import ApplicationServices
import CoreGraphics
import Foundation

// FERTIG's passive recorder deliberately listens only.  It never posts,
// modifies, or suppresses an input event.
private let stopKeyCode: Int64 = 100 // F8 on Apple's virtual-key-code layout.
private let allowUnverifiedText = CommandLine.arguments.contains("--allow-unverified-text")
private let checkOnly = CommandLine.arguments.contains("--check")
private var eventTap: CFMachPort?

private func emit(_ object: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(object),
          let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    else {
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

private func timestamp() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

if checkOnly {
    let inputMonitoring: Bool
    if #available(macOS 10.15, *) {
        inputMonitoring = CGPreflightListenEventAccess()
    } else {
        inputMonitoring = true
    }
    emit([
        "type": "status",
        "timestamp_ns": NSNumber(value: timestamp()),
        "status": "permission_check",
        "input_monitoring": inputMonitoring,
        "accessibility": AXIsProcessTrusted(),
        "input_sent": false,
    ])
    exit(0)
}

private func modifierNames(_ flags: CGEventFlags) -> [String] {
    var result: [String] = []
    if flags.contains(.maskShift) { result.append("shift") }
    if flags.contains(.maskControl) { result.append("control") }
    if flags.contains(.maskAlternate) { result.append("option") }
    if flags.contains(.maskCommand) { result.append("command") }
    if flags.contains(.maskAlphaShift) { result.append("caps_lock") }
    if flags.contains(.maskSecondaryFn) { result.append("function") }
    return result
}

// nil means that Accessibility could not prove whether the focused element is
// secure.  In that case text is suppressed unless the caller explicitly opted
// into --allow-unverified-text.  A secure field always suppresses text.
private func focusedElementIsSecure() -> Bool? {
    guard AXIsProcessTrusted() else { return nil }
    let system = AXUIElementCreateSystemWide()
    var focusedValue: CFTypeRef?
    guard AXUIElementCopyAttributeValue(
        system,
        kAXFocusedUIElementAttribute as CFString,
        &focusedValue
    ) == .success,
    let value = focusedValue,
    CFGetTypeID(value) == AXUIElementGetTypeID()
    else {
        return nil
    }

    let focused = unsafeBitCast(value, to: AXUIElement.self)
    var subroleValue: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(
        focused,
        kAXSubroleAttribute as CFString,
        &subroleValue
    )
    guard status == .success else { return nil }
    guard let subrole = subroleValue as? String else { return nil }
    return subrole == (kAXSecureTextFieldSubrole as String)
}

private func printableText(from event: CGEvent) -> String? {
    var units = [UniChar](repeating: 0, count: 64)
    var length = 0
    units.withUnsafeMutableBufferPointer { buffer in
        event.keyboardGetUnicodeString(
            maxStringLength: buffer.count,
            actualStringLength: &length,
            unicodeString: buffer.baseAddress
        )
    }
    guard length > 0 else { return nil }
    let text = String(utf16CodeUnits: units, count: length)
    guard !text.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
    else {
        return nil
    }
    return text
}

private func tapCallback(
    proxy _: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    refcon _: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        emit([
            "type": "status",
            "status": "tap_reenabled",
            "timestamp_ns": NSNumber(value: timestamp()),
        ])
        if let tap = eventTap { CGEvent.tapEnable(tap: tap, enable: true) }
        return Unmanaged.passUnretained(event)
    }

    let now = timestamp()
    let flags = modifierNames(event.flags)
    switch type {
    case .leftMouseDown, .rightMouseDown, .otherMouseDown,
         .leftMouseUp, .rightMouseUp, .otherMouseUp:
        let point = event.location
        let isDown = type == .leftMouseDown || type == .rightMouseDown || type == .otherMouseDown
        emit([
            "type": isDown ? "mouse_down" : "mouse_up",
            "timestamp_ns": NSNumber(value: now),
            "event_timestamp": NSNumber(value: event.timestamp),
            "x": point.x,
            "y": point.y,
            "button": NSNumber(value: event.getIntegerValueField(.mouseEventButtonNumber)),
            "click_state": NSNumber(value: event.getIntegerValueField(.mouseEventClickState)),
            "modifiers": flags,
        ])

    case .keyDown:
        let keycode = event.getIntegerValueField(.keyboardEventKeycode)
        if keycode == stopKeyCode {
            emit([
                "type": "stop",
                "timestamp_ns": NSNumber(value: now),
                "keycode": NSNumber(value: keycode),
                "modifiers": flags,
                "reason": "f8",
            ])
            CFRunLoopStop(CFRunLoopGetMain())
            return Unmanaged.passUnretained(event)
        }

        let secureState = focusedElementIsSecure()
        var payload: [String: Any] = [
            "type": "key_down",
            "timestamp_ns": NSNumber(value: now),
            "event_timestamp": NSNumber(value: event.timestamp),
            "keycode": NSNumber(value: keycode),
            "repeat": event.getIntegerValueField(.keyboardEventAutorepeat) != 0,
            "modifiers": flags,
        ]
        if secureState == true {
            payload["text_suppressed"] = "secure_text_field"
        } else if secureState == nil && !allowUnverifiedText {
            payload["text_suppressed"] = "secure_field_detection_unavailable"
        } else if let text = printableText(from: event) {
            payload["text"] = text
            payload["secure_focus_verified"] = secureState != nil
        }
        emit(payload)

    default:
        break
    }
    return Unmanaged.passUnretained(event)
}

if #available(macOS 10.15, *), !CGPreflightListenEventAccess() {
    emit([
        "type": "error",
        "timestamp_ns": NSNumber(value: timestamp()),
        "code": "input_monitoring_denied",
        "message": "Enable Input Monitoring for this terminal or app in System Settings > Privacy & Security.",
    ])
    exit(77)
}

let secureDetectionAvailable = AXIsProcessTrusted()
let eventTypes: [CGEventType] = [
    .leftMouseDown, .leftMouseUp,
    .rightMouseDown, .rightMouseUp,
    .otherMouseDown, .otherMouseUp,
    .keyDown,
]
let mask = eventTypes.reduce(CGEventMask(0)) {
    $0 | (CGEventMask(1) << CGEventMask($1.rawValue))
}

eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .listenOnly,
    eventsOfInterest: mask,
    callback: tapCallback,
    userInfo: nil
)

guard let tap = eventTap else {
    emit([
        "type": "error",
        "timestamp_ns": NSNumber(value: timestamp()),
        "code": "event_tap_unavailable",
        "message": "Could not create a passive CGEventTap; grant Input Monitoring and Accessibility.",
    ])
    exit(77)
}

guard let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0) else {
    emit([
        "type": "error",
        "timestamp_ns": NSNumber(value: timestamp()),
        "code": "run_loop_source_failed",
        "message": "Could not attach the passive event tap to a run loop.",
    ])
    exit(70)
}

CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
CGEvent.tapEnable(tap: tap, enable: true)
emit([
    "type": "ready",
    "timestamp_ns": NSNumber(value: timestamp()),
    "stop_key": "f8",
    "secure_text_detection": secureDetectionAvailable,
    "allow_unverified_text": allowUnverifiedText,
    "warning": secureDetectionAvailable
        ? "Text in AXSecureTextField elements is suppressed."
        : "Printable text is suppressed until Accessibility is granted, unless --allow-unverified-text is explicit.",
])
CFRunLoopRun()
emit([
    "type": "closed",
    "timestamp_ns": NSNumber(value: timestamp()),
    "reason": "run_loop_stopped",
])
