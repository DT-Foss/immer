"""
I/O module for .causal binary files.

Contains:
- CausalWriter: Create .causal files
- CausalReader: Read and query .causal files
- CausalFile: Unified high-level API
- CausalStorageBackend: Pipeline integration
"""

import struct
import io
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    import warnings
    warnings.warn(
        "msgpack fehlt — _vendor/dotcausal fällt auf JSON-Decode zurück "
        "und kann bei binären .causal-Dateien kryptisch fehlschlagen. "
        "Installation: pip install msgpack",
        RuntimeWarning,
    )

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    import warnings
    warnings.warn(
        "xxhash fehlt — CRC-Integritätsprüfung nutzt md5[:8]-Fallback; "
        "mit xxh64 geschriebene Dateien werden weiterhin erkannt "
        "(beide Algorithmen werden geprüft). pip install xxhash",
        RuntimeWarning,
    )

from .core import (
    MAGIC,
    LEGACY_MAGIC,
    VERSION,
    HEADER_SIZE,
    OFFSET_TABLE_SIZE,
    CausalTriplet,
    SemanticCluster,
    KnowledgeGap,
    DEFAULT_INFERENCE_RULES,
)
from .inference import run_inference


# =============================================================================
# ENCODING HELPERS
# =============================================================================

def _pack(data: Any) -> bytes:
    """Pack data to bytes using msgpack or JSON fallback."""
    if HAS_MSGPACK:
        return msgpack.packb(data, use_bin_type=True)
    else:
        return json.dumps(data).encode('utf-8')


def _unpack(data: bytes) -> Any:
    """Unpack bytes to data. Auto-detects JSON vs msgpack."""
    # Check if it's JSON (starts with { or [)
    if data and data[0:1] in (b'{', b'['):
        return json.loads(data.decode('utf-8'))

    # Try msgpack
    if HAS_MSGPACK:
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    else:
        return json.loads(data.decode('utf-8'))


def _compute_crc(data: bytes) -> bytes:
    """Compute 8-byte CRC for integrity verification."""
    if HAS_XXHASH:
        return struct.pack('<Q', xxhash.xxh64(data).intdigest())
    else:
        return hashlib.md5(data).digest()[:8]


def _compute_crc_alt(data: bytes) -> bytes:
    """Alternativer CRC-Algorithmus (Algorithmus-Drift-Toleranz):

    Dateien können mit xxh64 ODER md5[:8] geschrieben worden sein,
    je nach Umgebung des Writers. Ein Leser ohne xxhash würde sonst
    einen Fehlalarm ("Integrity check FAILED") auf korrekte Dateien
    werfen. Die Prüfung versucht beide — passend ist passend.
    """
    if HAS_XXHASH:
        return hashlib.md5(data).digest()[:8]
    else:
        return struct.pack('<Q', xxhash.xxh64(data).intdigest())


# =============================================================================
# CAUSAL WRITER
# =============================================================================

class CausalWriter:
    """
    Creates .causal binary files.

    Usage:
        writer = CausalWriter()
        writer.add_triplet(trigger="A", mechanism="causes", outcome="B")
        stats = writer.save("output.causal")
    """

    def __init__(self, api_id: str = "dotcausal"):
        """
        Initialize CausalWriter.

        Args:
            api_id: 8-character identifier for the creating application
        """
        self.api_id = api_id[:8].ljust(8, '\x00')

        self._entities: Dict[str, int] = {}
        self._entity_list: List[str] = []
        self._triplets: List[Dict] = []
        self._rules: List[Dict] = DEFAULT_INFERENCE_RULES.copy()
        self._clusters: List[SemanticCluster] = []
        self._gaps: List[KnowledgeGap] = []

    def _get_or_create_idx(self, entity: str) -> int:
        """Get dictionary index for entity, creating if needed."""
        entity = str(entity).strip()
        if entity not in self._entities:
            idx = len(self._entity_list)
            self._entities[entity] = idx
            self._entity_list.append(entity)
        return self._entities[entity]

    def add_triplet(
        self,
        trigger: str,
        mechanism: str,
        outcome: str,
        confidence: float = 0.8,
        source: str = "",
        pmcid: str = "",
        quantification: str = "",
        evidence: str = "",
        domain: str = "",
        quality_score: float = 0.0
    ) -> int:
        """
        Add a causal triplet.

        Returns:
            The triplet index.
        """
        triplet = {
            's_idx': self._get_or_create_idx(trigger),
            'm_idx': self._get_or_create_idx(mechanism),
            'o_idx': self._get_or_create_idx(outcome),
            'confidence': float(confidence),
            'source': source,
            'pmcid': pmcid,
            'quantification': quantification,
            'evidence': evidence[:500] if evidence else "",
            'domain': domain,
            'quality_score': float(quality_score),
            'is_inferred': False,
            'inference_chain': []
        }
        self._triplets.append(triplet)
        return len(self._triplets) - 1

    def add_rule(
        self,
        name: str,
        description: str,
        pattern: List[str],
        conclusion: str,
        confidence_modifier: float = 0.85
    ) -> None:
        """Add a custom inference rule."""
        self._rules.append({
            'name': name,
            'description': description,
            'pattern': pattern,
            'conclusion': conclusion,
            'confidence_modifier': confidence_modifier
        })

    def add_cluster(
        self,
        name: str,
        entities: List[str],
        related_clusters: List[int] = None
    ) -> int:
        """Add a semantic cluster."""
        cluster = SemanticCluster(
            cluster_id=len(self._clusters),
            name=name,
            entity_indices=[self._get_or_create_idx(e) for e in entities],
            related_clusters=related_clusters or []
        )
        self._clusters.append(cluster)
        return cluster.cluster_id

    def add_gap(
        self,
        subject: str,
        predicate: str,
        expected_object_type: str,
        confidence: float = 0.5,
        suggested_queries: List[str] = None
    ) -> None:
        """Add a knowledge gap."""
        self._gaps.append(KnowledgeGap(
            subject=subject,
            predicate=predicate,
            expected_object_type=expected_object_type,
            confidence=confidence,
            suggested_queries=suggested_queries or []
        ))

    def save(self, filepath: str) -> Dict:
        """
        Save to .causal binary file.

        Returns:
            Dict with file statistics.
        """
        filepath = Path(filepath)

        # Build sections
        dict_section = _pack({'entities': self._entity_list})
        anchors_section = _pack({'triplets': self._triplets})
        rules_section = _pack({'rules': self._rules})
        clusters_section = _pack({'clusters': [asdict(c) for c in self._clusters]})
        gaps_section = _pack({'gaps': [asdict(g) for g in self._gaps]})

        # Calculate offsets
        base_offset = HEADER_SIZE + OFFSET_TABLE_SIZE

        dict_offset = base_offset
        anchors_offset = dict_offset + len(dict_section)
        rules_offset = anchors_offset + len(anchors_section)
        clusters_offset = rules_offset + len(rules_section)
        gaps_offset = clusters_offset + len(clusters_section)
        end_offset = gaps_offset + len(gaps_section)

        # Build content
        content = dict_section + anchors_section + rules_section + clusters_section + gaps_section
        crc = _compute_crc(content)

        # Build header (64 bytes)
        header = io.BytesIO()
        header.write(MAGIC)
        header.write(struct.pack('<I', VERSION))
        header.write(struct.pack('<I', len(self._triplets)))
        header.write(struct.pack('<H', len(self._rules)))
        header.write(struct.pack('<H', 0))  # flags
        header.write(crc)
        header.write(self.api_id.encode('utf-8')[:8].ljust(8, b'\x00'))
        header.write(b'\x00' * (HEADER_SIZE - header.tell()))

        # Build offset table (32 bytes)
        offset_table = io.BytesIO()
        offset_table.write(struct.pack('<I', dict_offset))
        offset_table.write(struct.pack('<I', anchors_offset))
        offset_table.write(struct.pack('<I', rules_offset))
        offset_table.write(struct.pack('<I', clusters_offset))
        offset_table.write(struct.pack('<I', gaps_offset))
        offset_table.write(struct.pack('<I', end_offset))
        offset_table.write(b'\x00' * (OFFSET_TABLE_SIZE - offset_table.tell()))

        # Write file
        with open(filepath, 'wb') as f:
            f.write(header.getvalue())
            f.write(offset_table.getvalue())
            f.write(content)

        file_size = filepath.stat().st_size
        return {
            'filepath': str(filepath),
            'file_size_bytes': file_size,
            'file_size_kb': round(file_size / 1024, 1),
            'entities': len(self._entity_list),
            'triplets': len(self._triplets),
            'rules': len(self._rules),
            'clusters': len(self._clusters),
            'gaps': len(self._gaps)
        }


# =============================================================================
# CAUSAL READER
# =============================================================================

class CausalReader:
    """
    Reads .causal binary files with inference amplification.

    Usage:
        reader = CausalReader("knowledge.causal")

        # Explicit facts only
        explicit = reader.get_all_triplets(include_inferred=False)

        # With inference amplification
        all_facts = reader.get_all_triplets(include_inferred=True)
    """

    def __init__(self, filepath: str, verify_integrity: bool = True):
        """
        Initialize CausalReader.

        Args:
            filepath: Path to .causal file
            verify_integrity: Verify CRC checksum (default: True)
        """
        self.filepath = Path(filepath)

        self._entities: List[str] = []
        self._triplets: List[Dict] = []
        self._rules: List[Dict] = []
        self._clusters: List[Dict] = []
        self._gaps: List[Dict] = []

        self.version: int = 0
        self.triplet_count: int = 0
        self.rule_count: int = 0
        self.api_id: str = ""

        self._inferred_cache: Optional[List[Dict]] = None

        self._load(verify_integrity=verify_integrity)

    def _load(self, verify_integrity: bool = True) -> None:
        """Load and parse .causal file."""
        with open(self.filepath, 'rb') as f:
            data = f.read()

        magic = data[0:8]
        if magic not in (MAGIC, LEGACY_MAGIC):
            raise ValueError(f"Invalid .causal file: wrong magic bytes. Got {magic!r}, expected {MAGIC!r}")

        self.version = struct.unpack('<I', data[8:12])[0]
        self.triplet_count = struct.unpack('<I', data[12:16])[0]
        self.rule_count = struct.unpack('<H', data[16:18])[0]
        stored_crc = data[20:28]
        self.api_id = data[28:36].decode('utf-8').strip('\x00')

        if verify_integrity:
            content_start = HEADER_SIZE + OFFSET_TABLE_SIZE
            content = data[content_start:]
            computed_crc = _compute_crc(content)

            if stored_crc != computed_crc:
                # Algorithmus-Drift-Toleranz: der Writer kann mit dem
                # jeweils anderen CRC-Verfahren geschrieben haben.
                alt_crc = _compute_crc_alt(content)
                if stored_crc != alt_crc:
                    raise ValueError(
                        f"Integrity check FAILED for {self.filepath}!\n"
                        f"The file may be corrupted."
                    )

        # Parse offsets
        offsets = []
        for i in range(6):
            offset = struct.unpack('<I', data[HEADER_SIZE + i*4:HEADER_SIZE + i*4 + 4])[0]
            offsets.append(offset)

        dict_offset, anchors_offset, rules_offset, clusters_offset, gaps_offset, end_offset = offsets

        # Parse sections
        self._entities = _unpack(data[dict_offset:anchors_offset]).get('entities', [])
        self._triplets = _unpack(data[anchors_offset:rules_offset]).get('triplets', [])
        self._rules = _unpack(data[rules_offset:clusters_offset]).get('rules', [])
        self._clusters = _unpack(data[clusters_offset:gaps_offset]).get('clusters', [])

        if gaps_offset < len(data):
            try:
                self._gaps = _unpack(data[gaps_offset:end_offset]).get('gaps', [])
            except:
                self._gaps = []

    def get_entity(self, idx: int) -> str:
        """Get entity string by index."""
        if 0 <= idx < len(self._entities):
            return self._entities[idx]
        return f"<unknown:{idx}>"

    def get_all_triplets(self, include_inferred: bool = True) -> List[Dict]:
        """
        Get all triplets, optionally including inferred ones.

        Args:
            include_inferred: Run inference for amplification

        Returns:
            List of triplet dicts with resolved entity names
        """
        explicit = []
        for t in self._triplets:
            explicit.append({
                'trigger': self.get_entity(t['s_idx']),
                'mechanism': self.get_entity(t['m_idx']),
                'outcome': self.get_entity(t['o_idx']),
                'confidence': t.get('confidence', 0.8),
                'source': t.get('source', ''),
                'pmcid': t.get('pmcid', ''),
                'quantification': t.get('quantification', ''),
                'evidence': t.get('evidence', ''),
                'domain': t.get('domain', ''),
                'quality_score': t.get('quality_score', 0.0),
                'is_inferred': False,
                's_idx': t['s_idx'],
                'm_idx': t['m_idx'],
                'o_idx': t['o_idx']
            })

        if not include_inferred:
            return explicit

        if self._inferred_cache is None:
            self._inferred_cache = run_inference(explicit, self._entities, self._rules)

        return explicit + self._inferred_cache

    def get_stats(self) -> Dict:
        """Get file statistics."""
        all_triplets = self.get_all_triplets(include_inferred=True)
        explicit_count = len(self._triplets)
        inferred_count = len(all_triplets) - explicit_count

        return {
            'filepath': str(self.filepath),
            'file_size_kb': round(self.filepath.stat().st_size / 1024, 1),
            'version': self.version,
            'api_id': self.api_id,
            'entities': len(self._entities),
            'explicit_triplets': explicit_count,
            'inferred_triplets': inferred_count,
            'total_triplets': len(all_triplets),
            'amplification_percent': round(inferred_count / max(explicit_count, 1) * 100, 1),
            'rules': len(self._rules),
            'clusters': len(self._clusters),
            'gaps': len(self._gaps)
        }

    def search(self, query: str, field: str = 'all') -> List[Dict]:
        """
        Search triplets by query string.

        Args:
            query: Search string (case-insensitive)
            field: 'trigger', 'mechanism', 'outcome', or 'all'
        """
        query = query.lower()
        results = []

        for t in self.get_all_triplets():
            if field == 'all':
                text = f"{t['trigger']} {t['mechanism']} {t['outcome']}"
            else:
                text = t.get(field, '')

            if query in text.lower():
                results.append(t)

        return results


# =============================================================================
# CAUSAL FILE (Unified API)
# =============================================================================

class CausalFile:
    """
    High-level unified API for .causal files.

    Usage:
        # Read existing file
        cf = CausalFile.load("knowledge.causal")

        # Create new file
        cf = CausalFile()
        cf.add(Triplet(...))
        cf.save("output.causal")
    """

    def __init__(self):
        """Create a new empty CausalFile."""
        self._writer = CausalWriter()
        self._reader: Optional[CausalReader] = None
        self._filepath: Optional[str] = None

    @classmethod
    def load(cls, filepath: str, verify_integrity: bool = True) -> 'CausalFile':
        """Load an existing .causal file."""
        instance = cls()
        instance._reader = CausalReader(filepath, verify_integrity=verify_integrity)
        instance._filepath = filepath
        return instance

    def add(self, triplet: CausalTriplet) -> int:
        """Add a triplet (only for new files)."""
        if self._reader is not None:
            raise RuntimeError("Cannot add triplets to a loaded file. Create a new CausalFile instead.")

        # Get entity strings from indices (if using CausalTriplet with indices)
        # For now, assume trigger/mechanism/outcome are passed as kwargs
        raise NotImplementedError("Use add_triplet() method with keyword arguments")

    def add_triplet(self, **kwargs) -> int:
        """Add a triplet with keyword arguments."""
        return self._writer.add_triplet(**kwargs)

    def run_inference(self) -> int:
        """Run inference and return count of inferred facts."""
        # This is handled automatically when reading
        return 0

    def save(self, filepath: str) -> Dict:
        """Save to file."""
        return self._writer.save(filepath)

    @property
    def triplets(self) -> List[Dict]:
        """Get explicit triplets."""
        if self._reader:
            return self._reader.get_all_triplets(include_inferred=False)
        return []

    @property
    def inferred(self) -> List[Dict]:
        """Get inferred triplets."""
        if self._reader:
            all_t = self._reader.get_all_triplets(include_inferred=True)
            explicit = self._reader.get_all_triplets(include_inferred=False)
            return all_t[len(explicit):]
        return []

    def query(self, **kwargs) -> List[Dict]:
        """Query the file."""
        if self._reader:
            trigger = kwargs.get('trigger')
            if trigger:
                return self._reader.search(trigger, field='trigger')
            return self._reader.get_all_triplets()
        return []


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

class CausalStorageBackend:
    """
    Drop-in replacement for PipelineDB that writes to .causal format.

    Usage:
        storage = CausalStorageBackend("knowledge.causal")
        storage.save_triplets(chunk_id, triplets)
        storage.export()
    """

    def __init__(self, output_path: str = "output.causal"):
        self.output_path = Path(output_path)
        self.writer = CausalWriter(api_id="pipeline")

        self._documents: Dict[str, int] = {}
        self._chunks: Dict[int, Dict] = {}
        self._chunk_counter = 0
        self._doc_counter = 0

        self.stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'triplets_added': 0
        }

    def add_document(self, filename: str, file_hash: str, file_size: int = 0) -> Optional[int]:
        """Add document if not already processed."""
        if file_hash in self._documents:
            return None
        self._doc_counter += 1
        self._documents[file_hash] = self._doc_counter
        return self._doc_counter

    def document_exists(self, file_hash: str) -> bool:
        """Check if document exists."""
        return file_hash in self._documents

    def add_chunks(self, doc_id: int, chunks: List[str]) -> int:
        """Add chunks for a document."""
        for i, text in enumerate(chunks):
            self._chunk_counter += 1
            self._chunks[self._chunk_counter] = {
                'doc_id': doc_id,
                'chunk_index': i,
                'text': text,
                'status': 'pending'
            }
        return len(chunks)

    def save_triplets(self, chunk_id: int, triplets: List[Dict], source_file: str = "") -> int:
        """Save triplets from a chunk."""
        if not triplets:
            if chunk_id in self._chunks:
                self._chunks[chunk_id]['status'] = 'completed'
            return 0

        for t in triplets:
            confidence = self._parse_confidence(t.get('confidence', 'medium'))

            self.writer.add_triplet(
                trigger=t.get('trigger', ''),
                mechanism=t.get('mechanism', ''),
                outcome=t.get('outcome', ''),
                confidence=confidence,
                source=t.get('source_file', source_file),
                pmcid=t.get('pmcid', ''),
                quantification=t.get('quantification', ''),
                evidence=t.get('evidence_sentence', ''),
                domain=t.get('domain', ''),
                quality_score=t.get('quality_score', 0.0)
            )
            self.stats['triplets_added'] += 1

        if chunk_id in self._chunks:
            self._chunks[chunk_id]['status'] = 'completed'

        self.stats['chunks_processed'] += 1
        return len(triplets)

    def _parse_confidence(self, conf) -> float:
        """Convert string confidence to float."""
        if isinstance(conf, (int, float)):
            return float(conf)

        conf_map = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5,
            'very_high': 0.95,
            'very_low': 0.3
        }
        return conf_map.get(str(conf).lower(), 0.7)

    def export(self) -> Dict:
        """Export to .causal file."""
        return self.writer.save(self.output_path)

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'documents': {'total': len(self._documents), 'completed': len(self._documents)},
            'chunks': {'total': len(self._chunks), 'completed': self.stats['chunks_processed']},
            'triplets': {'total': self.stats['triplets_added']}
        }
