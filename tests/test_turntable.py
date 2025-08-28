"""
Turntable Module Tests
======================

Comprehensive unit test suite for the Turntable interface module. This
test suite validates all functionality of the Turntable class and its
associated exception classes using mock objects to simulate hardware
interactions without requiring physical device connections.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import serial
import unittest
from unittest.mock import Mock, patch
from micromanipulator_tools import (
    Turntable,
    TurntableError,
    TurntableConnectionError,
    TurntableCommandError,
    TurntableTimeoutError,
)

__all__ = [
    "Turntable",
    "TurntableError",
    "TurntableConnectionError",
    "TurntableCommandError",
    "TurntableTimeoutError",
]


class TestTurntable(unittest.TestCase):
    """
    Comprehensive unit test suite for the Turntable stepper motor
    controller.

    Tests all functionality of the Turntable class and its associated
    exception classes using mock objects to simulate Arduino hardware
    interactions without requiring physical device connections.

    Tests cover initialisation, context management, serial
    communication, command building, response parsing, and the main
    rotate() interface with comprehensive error handling validation.
    """

    def test_init_valid_parameters(self):
        """
        Test __init__ with valid parameters and successful connection.
        """
        mock_serial = Mock()
        mock_serial.read_until.return_value = (
            b"\\NAK: 2: Error while parsing command: E\r"
        )
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            tt = Turntable("COM3", 9600, 30)

            # Verify attributes are set
            self.assertEqual(tt.com_port, "COM3")
            self.assertEqual(tt.baud_rate, 9600)
            self.assertEqual(tt.timeout_sec, 30)
            self.assertEqual(tt.device_serial, mock_serial)

    def test_init_default_parameters(self):
        """
        Test __init__ with default baud_rate and timeout_sec.
        """
        mock_serial = Mock()
        mock_serial.read_until.return_value = (
            b"\\NAK: 2: Error while parsing command: E\r"
        )
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch(
            "serial.Serial", return_value=mock_serial
        ) as mock_serial_constructor:
            tt = Turntable("COM3")

            # Verify defaults were used
            self.assertEqual(tt.baud_rate, 9600)
            self.assertEqual(tt.timeout_sec, 30)

            # Verify serial.Serial was called with defaults
            mock_serial_constructor.assert_called_once_with(
                "COM3", 9600, timeout=30
            )

    def test_init_invalid_com_port_empty(self):
        """
        Test __init__ raises ValueError for empty com_port.
        """
        with self.assertRaises(ValueError) as cm:
            Turntable("", 9600, 30)

        self.assertIn("com_port must be a non-empty string", str(cm.exception))

    def test_init_invalid_com_port_whitespace(self):
        """
        Test __init__ raises ValueError for whitespace-only com_port.
        """
        with self.assertRaises(ValueError) as cm:
            Turntable("   ", 9600, 30)

        self.assertIn("com_port must be a non-empty string", str(cm.exception))

    def test_init_invalid_com_port_type(self):
        """
        Test __init__ raises ValueError for non-string com_port.
        """
        invalid_ports = [123, None, ["COM3"], {"port": "COM3"}]

        for invalid_port in invalid_ports:
            with self.subTest(port=invalid_port):
                with self.assertRaises(ValueError) as cm:
                    Turntable(invalid_port, 9600, 30)

                self.assertIn(
                    "com_port must be a non-empty string", str(cm.exception)
                )

    def test_init_invalid_baud_rate_type(self):
        """
        Test __init__ raises ValueError for non-integer baud_rate.
        """
        invalid_baud_rates = [9600.0, "9600", None, [9600]]

        for invalid_baud in invalid_baud_rates:
            with self.subTest(baud=invalid_baud):
                with self.assertRaises(ValueError) as cm:
                    Turntable("COM3", invalid_baud, 30)

                self.assertIn(
                    "baud_rate must be a positive integer", str(cm.exception)
                )

    def test_init_invalid_baud_rate_value(self):
        """
        Test __init__ raises ValueError for non-positive baud_rate.
        """
        invalid_baud_rates = [0, -1, -9600]

        for invalid_baud in invalid_baud_rates:
            with self.subTest(baud=invalid_baud):
                with self.assertRaises(ValueError) as cm:
                    Turntable("COM3", invalid_baud, 30)

                self.assertIn(
                    "baud_rate must be a positive integer", str(cm.exception)
                )

    def test_init_invalid_timeout_type(self):
        """
        Test __init__ raises ValueError for invalid timeout_sec types.
        """
        invalid_timeouts = ["30", None, [30], {"timeout": 30}]

        for invalid_timeout in invalid_timeouts:
            with self.subTest(timeout=invalid_timeout):
                with self.assertRaises(ValueError) as cm:
                    Turntable("COM3", 9600, invalid_timeout)

                self.assertIn(
                    "timeout_sec must be a positive number", str(cm.exception)
                )

    def test_init_invalid_timeout_value(self):
        """
        Test __init__ raises ValueError for non-positive timeout_sec.
        """
        invalid_timeouts = [0, -1, -30.5]

        for invalid_timeout in invalid_timeouts:
            with self.subTest(timeout=invalid_timeout):
                with self.assertRaises(ValueError) as cm:
                    Turntable("COM3", 9600, invalid_timeout)

                self.assertIn(
                    "timeout_sec must be a positive number", str(cm.exception)
                )

    def test_init_serial_connection_failure(self):
        """
        Test __init__ raises TurntableConnectionError when serial connection fails.
        """
        with patch(
            "serial.Serial",
            side_effect=serial.SerialException("Port not found"),
        ):
            with self.assertRaises(TurntableConnectionError) as cm:
                Turntable("COM3")

            error_msg = str(cm.exception)
            self.assertIn("Failed to connect to turntable on COM3", error_msg)
            self.assertIn("Port not found", error_msg)

    def test_init_handshake_failure(self):
        """
        Test __init__ raises TurntableConnectionError when handshake fails.
        """
        mock_serial = Mock()
        # Wrong response (should start with "2:")
        mock_serial.read_until.return_value = b"\\ACK: Unexpected response\r"
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            with self.assertRaises(TurntableConnectionError) as cm:
                Turntable("COM3")

            error_msg = str(cm.exception)
            self.assertIn("Handshake failed", error_msg)
            self.assertIn("Expected syntax error", error_msg)

    def test_init_closes_serial_on_failure(self):
        """
        Test __init__ closes serial port when initialisation fails.
        """
        mock_serial = Mock()
        mock_serial.read_until.return_value = b"\\ACK: Wrong response\r"
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()
        mock_serial.is_open = True

        with patch("serial.Serial", return_value=mock_serial):
            with patch.object(Turntable, "_close_serial") as mock_close:
                with self.assertRaises(TurntableConnectionError):
                    Turntable("COM3")

                # Should call _close_serial on failure
                mock_close.assert_called_once()

    def test_init_successful_handshake(self):
        """
        Test __init__ successful handshake with expected syntax error response.
        """
        mock_serial = Mock()
        mock_serial.read_until.return_value = (
            b"\\NAK: 2: Error while parsing command: E\r"
        )
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            with patch.object(
                Turntable,
                "_send_command",
                return_value="\\NAK: 2: Error while parsing command: E",
            ):
                with patch.object(
                    Turntable,
                    "_parse_response",
                    return_value="2: Error while parsing command: E",
                ):
                    tt = Turntable("COM3")

                    # Should succeed without raising exception
                    self.assertIsInstance(tt, Turntable)

    def test_init_serial_operations_sequence(self):
        """
        Test __init__ calls serial operations in correct sequence.
        """
        call_order = []

        def track_flush_input():
            call_order.append("flushInput")

        def track_send_command(cmd):
            call_order.append(("send_command", cmd))
            return "\\NAK: 2: Syntax error"

        def track_parse_response(response):
            call_order.append(("parse_response", response))
            return "2: Syntax error"

        mock_serial = Mock()
        mock_serial.flushInput.side_effect = track_flush_input

        with patch("serial.Serial", return_value=mock_serial):
            with patch.object(
                Turntable, "_send_command", side_effect=track_send_command
            ):
                with patch.object(
                    Turntable,
                    "_parse_response",
                    side_effect=track_parse_response,
                ):
                    Turntable("COM3")

                    expected_sequence = [
                        "flushInput",
                        ("send_command", "E"),
                        ("parse_response", "\\NAK: 2: Syntax error"),
                    ]

                    self.assertEqual(call_order, expected_sequence)

    def test_init_timeout_sec_accepts_float(self):
        """
        Test __init__ accepts float values for timeout_sec.
        """
        mock_serial = Mock()
        mock_serial.read_until.return_value = (
            b"\\NAK: 2: Error while parsing command: E\r"
        )
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            with patch.object(
                Turntable, "_send_command", return_value="\\NAK: 2: Error"
            ):
                with patch.object(
                    Turntable, "_parse_response", return_value="2: Error"
                ):
                    tt = Turntable("COM3", 9600, 30.5)

                    self.assertEqual(tt.timeout_sec, 30.5)

    def test_enter_returns_self(self):
        """
        Test that __enter__ returns the Turntable instance itself for
        use in context manager 'with' statements.
        """
        tt = Turntable.__new__(Turntable)

        result = tt.__enter__()

        self.assertIs(result, tt)

    def test_enter_context_manager_integration(self):
        """
        Test __enter__ works correctly in actual 'with' statement.
        """
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial.read_until.return_value = (
            b"\\NAK: 2: Error while parsing command: E\r"
        )
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            with Turntable("COM3") as tt:
                self.assertIsInstance(tt, Turntable)

    def test_exit_calls_close_serial(self):
        """
        Test that __exit__ calls _close_serial to clean up resources.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(tt, "_close_serial") as mock_close:
            tt.__exit__(None, None, None)

            mock_close.assert_called_once()

    def test_exit_logs_exception_info(self):
        """
        Test that __exit__ prints exception details when an exception occurs.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(tt, "_close_serial"):
            with patch("builtins.print") as mock_print:
                # Simulate exception occurring in context
                exception_type = ValueError
                exception_value = ValueError("test error")
                exception_traceback = "mock_traceback"

                tt.__exit__(
                    exception_type, exception_value, exception_traceback
                )

                # Should print all three exception details
                mock_print.assert_any_call(
                    "Turntable.__exit__: Exception type: <class 'ValueError'>"
                )
                mock_print.assert_any_call(
                    "Turntable.__exit__: Exception value: test error"
                )
                mock_print.assert_any_call(
                    "Turntable.__exit__: Traceback: mock_traceback"
                )

    def test_exit_no_logging_when_no_exception(self):
        """
        Test that __exit__ doesn't print anything when no exception occurs.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(tt, "_close_serial"):
            with patch("builtins.print") as mock_print:
                # No exception (normal exit)
                tt.__exit__(None, None, None)

                # Should not print anything
                mock_print.assert_not_called()

    def test_exit_returns_none(self):
        """
        Test that __exit__ returns None (doesn't suppress exceptions).
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(tt, "_close_serial"):
            # Test with no exception
            result = tt.__exit__(None, None, None)
            self.assertIsNone(result)

            # Test with exception
            result = tt.__exit__(ValueError, ValueError("test"), "traceback")
            self.assertIsNone(result)

    def test_str_returns_expected_format(self):
        """
        Test that __str__ returns the expected string format.
        """
        tt = Turntable.__new__(Turntable)
        tt.com_port = "COM3"

        result = str(tt)

        self.assertEqual(result, "Turntable on COM3")

    def test_str_with_different_com_ports(self):
        """
        Test __str__ with various COM port values.
        """
        test_ports = ["COM1", "COM17", "/dev/ttyUSB0", "/dev/tty.usbserial"]

        for port in test_ports:
            with self.subTest(port=port):
                tt = Turntable.__new__(Turntable)
                tt.com_port = port

                result = str(tt)

                self.assertEqual(result, f"Turntable on {port}")

    def test_str_returns_string_type(self):
        """
        Test that __str__ returns a string type.
        """
        tt = Turntable.__new__(Turntable)
        tt.com_port = "COM3"

        result = str(tt)

        self.assertIsInstance(result, str)

    def test_close_serial_closes_open_port(self):
        """
        Test that _close_serial closes an open serial port.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.is_open = True
        tt.device_serial = mock_serial

        with patch("builtins.print") as mock_print:
            tt._close_serial()

        mock_serial.close.assert_called_once()
        mock_print.assert_called_with(
            "Turntable._close_serial: Serial connection closed."
        )

    def test_close_serial_does_nothing_when_port_closed(self):
        """
        Test that _close_serial does nothing when port is already closed.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.is_open = False
        tt.device_serial = mock_serial

        with patch("builtins.print") as mock_print:
            tt._close_serial()

        mock_serial.close.assert_not_called()
        mock_print.assert_not_called()

    def test_close_serial_does_nothing_when_no_device_serial(self):
        """
        Test that _close_serial does nothing when device_serial attribute doesn't exist.
        """
        tt = Turntable.__new__(Turntable)
        # Don't set device_serial attribute

        with patch("builtins.print") as mock_print:
            tt._close_serial()

        mock_print.assert_not_called()

    def test_close_serial_handles_close_exception(self):
        """
        Test that _close_serial handles exceptions during close operation.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial.close.side_effect = Exception("Close failed")
        tt.device_serial = mock_serial

        with patch("builtins.print") as mock_print:
            tt._close_serial()  # Should not raise exception

        mock_serial.close.assert_called_once()
        mock_print.assert_called_with(
            "Turntable._close_serial: WARNING: Exception when closing "
            "serial port: Close failed"
        )

    def test_close_serial_handles_device_serial_none(self):
        """
        Test that _close_serial handles when device_serial is None.
        """
        tt = Turntable.__new__(Turntable)
        tt.device_serial = None

        with patch("builtins.print") as mock_print:
            tt._close_serial()  # Should not raise exception

        # Should print warning about AttributeError
        mock_print.assert_called_with(
            "Turntable._close_serial: WARNING: Exception when closing "
            "serial port: 'NoneType' object has no attribute 'is_open'"
        )

    def test_build_command_valid_inputs(self):
        """
        Test _build_command with valid inputs generates correct command strings.
        """
        tt = Turntable.__new__(Turntable)

        test_cases = [
            # (speed, angle, reverse, expected_command)
            ("fast", 90.0, False, "M390.00E"),
            ("medium", 45.5, False, "M245.50E"),
            ("slow", 180.0, False, "M1180.00E"),
            ("fast", 90.0, True, "M3-90.00E"),
            ("medium", 45.5, True, "M2-45.50E"),
            ("slow", 180.0, True, "M1-180.00E"),
            ("fast", 0.0, False, "M30.00E"),
            ("fast", 360.0, False, "M3360.00E"),
        ]

        for speed, angle, reverse, expected in test_cases:
            with self.subTest(speed=speed, angle=angle, reverse=reverse):
                result = tt._build_command(speed, angle, reverse)
                self.assertEqual(result, expected)

    def test_build_command_invalid_angle_type(self):
        """
        Test _build_command raises ValueError for non-numeric angle.
        """
        tt = Turntable.__new__(Turntable)

        invalid_angles = ["90", None, [90], {"angle": 90}]

        for invalid_angle in invalid_angles:
            with self.subTest(angle=invalid_angle):
                with self.assertRaises(ValueError) as cm:
                    tt._build_command("fast", invalid_angle, False)

                self.assertIn(
                    "'angle_deg' must be a number", str(cm.exception)
                )

    def test_build_command_angle_out_of_range(self):
        """
        Test _build_command raises ValueError for angles outside 0-360 range.
        """
        tt = Turntable.__new__(Turntable)

        invalid_angles = [-0.1, -10, 360.1, 400, 720]

        for invalid_angle in invalid_angles:
            with self.subTest(angle=invalid_angle):
                with self.assertRaises(ValueError) as cm:
                    tt._build_command("fast", invalid_angle, False)

                error_msg = str(cm.exception)
                self.assertIn("'angle_deg' must be between", error_msg)
                self.assertIn("0.0 and 360.0 degrees", error_msg)

    def test_build_command_invalid_speed_type(self):
        """
        Test _build_command raises ValueError for non-string speed.
        """
        tt = Turntable.__new__(Turntable)

        invalid_speeds = [1, 1.5, None, ["fast"], {"speed": "fast"}]

        for invalid_speed in invalid_speeds:
            with self.subTest(speed=invalid_speed):
                with self.assertRaises(ValueError) as cm:
                    tt._build_command(invalid_speed, 90.0, False)

                self.assertIn("'speed' must be a string", str(cm.exception))

    def test_build_command_invalid_speed_value(self):
        """
        Test _build_command raises ValueError for invalid speed strings.
        """
        tt = Turntable.__new__(Turntable)

        invalid_speeds = ["", "turbo", "very_fast", "1", "2", "3"]

        for invalid_speed in invalid_speeds:
            with self.subTest(speed=invalid_speed):
                with self.assertRaises(ValueError) as cm:
                    tt._build_command(invalid_speed, 90.0, False)

                error_msg = str(cm.exception)
                self.assertIn("'speed' must be one of", error_msg)
                self.assertIn("['slow', 'medium', 'fast']", error_msg)

    def test_build_command_invalid_reverse_type(self):
        """
        Test _build_command raises ValueError for non-boolean reverse.
        """
        tt = Turntable.__new__(Turntable)

        invalid_reverse_values = [1, 0, "True", "False", None, [True]]

        for invalid_reverse in invalid_reverse_values:
            with self.subTest(reverse=invalid_reverse):
                with self.assertRaises(ValueError) as cm:
                    tt._build_command("fast", 90.0, invalid_reverse)

                self.assertIn("'reverse' must be a bool", str(cm.exception))

    def test_build_command_boundary_angles(self):
        """
        Test _build_command with boundary angle values.
        """
        tt = Turntable.__new__(Turntable)

        boundary_cases = [
            (0.0, False, "M30.00E"),
            (360.0, False, "M3360.00E"),
            (0.0, True, "M3-0.00E"),
            (360.0, True, "M3-360.00E"),
        ]

        for angle, reverse, expected in boundary_cases:
            with self.subTest(angle=angle, reverse=reverse):
                result = tt._build_command("fast", angle, reverse)
                self.assertEqual(result, expected)

    def test_send_command_successful_response(self):
        """
        Test _send_command with successful serial communication.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.read_until.return_value = b"\\ACK: Command executed\r"
        tt.device_serial = mock_serial

        result = tt._send_command("M390.00E")

        # Verify serial operations
        mock_serial.flushInput.assert_called_once()
        mock_serial.write.assert_called_once_with(b"M390.00E")
        mock_serial.flush.assert_called_once()
        mock_serial.read_until.assert_called_once_with(b"\r\n")

        self.assertEqual(result, "\\ACK: Command executed")

    def test_send_command_timeout_empty_response(self):
        """
        Test _send_command raises TurntableTimeoutError for empty response.
        """
        tt = Turntable.__new__(Turntable)
        tt.timeout_sec = 30

        mock_serial = Mock()
        mock_serial.read_until.return_value = b""  # Empty response (timeout)
        tt.device_serial = mock_serial

        with self.assertRaises(TurntableTimeoutError) as cm:
            tt._send_command("M390.00E")

        error_msg = str(cm.exception)
        self.assertIn("No response received within 30 seconds", error_msg)
        self.assertIn("M390.00E", error_msg)

    def test_send_command_unicode_encode_error(self):
        """
        Test _send_command raises ValueError for non-ASCII characters.
        """
        tt = Turntable.__new__(Turntable)

        # Command with non-ASCII character (°)
        invalid_command = "M390.00°E"

        with self.assertRaises(ValueError) as cm:
            tt._send_command(invalid_command)

        error_msg = str(cm.exception)
        self.assertIn("Command contains non-ASCII character", error_msg)
        self.assertIsInstance(cm.exception.__cause__, UnicodeEncodeError)

    def test_send_command_serial_exception(self):
        """
        Test _send_command raises TurntableConnectionError for serial errors.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.write.side_effect = serial.SerialException("Port closed")
        tt.device_serial = mock_serial

        with self.assertRaises(TurntableConnectionError) as cm:
            tt._send_command("M390.00E")

        error_msg = str(cm.exception)
        self.assertIn("Serial communication error", error_msg)
        self.assertIn("Port closed", error_msg)

    def test_send_command_strips_whitespace(self):
        """
        Test _send_command strips whitespace from response.
        """
        tt = Turntable.__new__(Turntable)

        mock_serial = Mock()
        mock_serial.read_until.return_value = (
            b"  \\ACK: Command executed  \r\n"
        )
        tt.device_serial = mock_serial

        result = tt._send_command("M390.00E")

        self.assertEqual(result, "\\ACK: Command executed")

    def test_send_command_various_responses(self):
        """
        Test _send_command with various valid response formats.
        """
        tt = Turntable.__new__(Turntable)

        test_responses = [
            b"\\ACK: Success\r",
            b"\\NAK: 2: Syntax error\r",
            b"\\ACK: Rotation complete\r",
            b"\\NAK: 1: Invalid message\r",
        ]

        for response_bytes in test_responses:
            with self.subTest(response=response_bytes):
                mock_serial = Mock()
                mock_serial.read_until.return_value = response_bytes
                tt.device_serial = mock_serial

                result = tt._send_command("M390.00E")

                expected = response_bytes.decode("ascii").strip()
                self.assertEqual(result, expected)

    def test_send_command_serial_operations_order(self):
        """
        Test that _send_command calls serial operations in correct order.
        """
        tt = Turntable.__new__(Turntable)

        call_order = []

        def track_flush_input():
            call_order.append("flushInput")

        def track_write(data):
            call_order.append(("write", data))

        def track_flush():
            call_order.append("flush")

        def track_read_until(delimiter):
            call_order.append(("read_until", delimiter))
            return b"\\ACK: OK\r"

        mock_serial = Mock()
        mock_serial.flushInput.side_effect = track_flush_input
        mock_serial.write.side_effect = track_write
        mock_serial.flush.side_effect = track_flush
        mock_serial.read_until.side_effect = track_read_until
        tt.device_serial = mock_serial

        tt._send_command("M390.00E")

        expected_order = [
            "flushInput",
            ("write", b"M390.00E"),
            "flush",
            ("read_until", b"\r\n"),
        ]

        self.assertEqual(call_order, expected_order)

    def test_send_command_different_serial_exceptions(self):
        """
        Test _send_command handles various serial exception types.
        """
        tt = Turntable.__new__(Turntable)

        serial_errors = [
            serial.SerialException("Port not found"),
            serial.SerialTimeoutException("Write timeout"),
            serial.SerialException("Device disconnected"),
        ]

        for error in serial_errors:
            with self.subTest(error=error):
                mock_serial = Mock()
                mock_serial.write.side_effect = error
                tt.device_serial = mock_serial

                with self.assertRaises(TurntableConnectionError) as cm:
                    tt._send_command("M390.00E")

                self.assertIn("Serial communication error", str(cm.exception))

    def test_parse_response_ack_success(self):
        """
        Test _parse_response with ACK responses returns message.
        """
        tt = Turntable.__new__(Turntable)

        ack_responses = [
            "\\ACK: Finished executing command: M390.00E",
            "\\ACK: Rotation complete",
            "\\ACK: Command received",
            "\\ACK: OK",
        ]

        for response in ack_responses:
            with self.subTest(response=response):
                result = tt._parse_response(response)

                expected = response[len(tt.ACK_PREFIX) :].strip()
                self.assertEqual(result, expected)

    def test_parse_response_nak_safe_errors(self):
        """
        Test _parse_response with NAK safe errors (prints warning, returns message).
        """
        tt = Turntable.__new__(Turntable)

        safe_error_responses = [
            "\\NAK: 1: Invalid message received",
            "\\NAK: 2: Error while parsing command: E",
        ]

        for response in safe_error_responses:
            with self.subTest(response=response):
                with patch("builtins.print") as mock_print:
                    result = tt._parse_response(response)

                    expected_message = response[len(tt.NAK_PREFIX) :].strip()
                    self.assertEqual(result, expected_message)

                    # Should print warning
                    mock_print.assert_called_once_with(
                        f"Turntable._parse_response: {expected_message}"
                    )

    def test_parse_response_nak_timeout_error(self):
        """
        Test _parse_response with NAK timeout error raises TurntableTimeoutError.
        """
        tt = Turntable.__new__(Turntable)

        timeout_response = "\\NAK: 3: Arduino operation timed out"

        with self.assertRaises(TurntableTimeoutError) as cm:
            tt._parse_response(timeout_response)

        error_msg = str(cm.exception)
        self.assertIn("Arduino timed out", error_msg)
        self.assertIn("3: Arduino operation timed out", error_msg)

    def test_parse_response_nak_unknown_error(self):
        """
        Test _parse_response with unknown NAK error code raises TurntableCommandError.
        """
        tt = Turntable.__new__(Turntable)

        unknown_error_responses = [
            "\\NAK: 4: Unknown error",
            "\\NAK: 9: Hardware failure",
            "\\NAK: 5: Motor stuck",
        ]

        for response in unknown_error_responses:
            with self.subTest(response=response):
                with self.assertRaises(TurntableCommandError) as cm:
                    tt._parse_response(response)

                error_msg = str(cm.exception)
                self.assertIn("Unknown Arduino error", error_msg)
                expected_message = response[len(tt.NAK_PREFIX) :].strip()
                self.assertIn(expected_message, error_msg)

    def test_parse_response_invalid_empty_response(self):
        """
        Test _parse_response with invalid or empty responses.
        """
        tt = Turntable.__new__(Turntable)

        # These will trigger "Invalid or empty response"
        truly_empty_responses = ["", None]

        for response in truly_empty_responses:
            with self.subTest(response=repr(response)):
                with self.assertRaises(TurntableCommandError) as cm:
                    tt._parse_response(response)

                self.assertIn("Invalid or empty response", str(cm.exception))

        # These will trigger "Unknown response format" (whitespace-only strings)
        whitespace_only_responses = ["   ", "\t\n"]

        for response in whitespace_only_responses:
            with self.subTest(response=repr(response)):
                with self.assertRaises(TurntableCommandError) as cm:
                    tt._parse_response(response)

                self.assertIn("Unknown response format", str(cm.exception))

    def test_parse_response_invalid_response_type(self):
        """
        Test _parse_response with non-string response types.
        """
        tt = Turntable.__new__(Turntable)

        invalid_types = [123, 45.6, ["response"], {"msg": "response"}]

        for response in invalid_types:
            with self.subTest(response=response):
                with self.assertRaises(TurntableCommandError) as cm:
                    tt._parse_response(response)

                self.assertIn("Invalid or empty response", str(cm.exception))

    def test_parse_response_unknown_format(self):
        """
        Test _parse_response with responses that don't start with ACK or NAK.
        """
        tt = Turntable.__new__(Turntable)

        unknown_formats = [
            "OK: Command complete",
            "ERROR: Something went wrong",
            "Random response text",
            "\\UNKNOWN: Weird format",
        ]

        for response in unknown_formats:
            with self.subTest(response=response):
                with self.assertRaises(TurntableCommandError) as cm:
                    tt._parse_response(response)

                error_msg = str(cm.exception)
                self.assertIn("Unknown response format", error_msg)
                self.assertIn(response, error_msg)

    def test_parse_response_error_code_extraction(self):
        """
        Test _parse_response correctly extracts error codes from NAK responses.
        """
        tt = Turntable.__new__(Turntable)

        # Test each known error code
        error_code_tests = [
            ("\\NAK: 1: Invalid message", 1, "safe"),
            ("\\NAK: 2: Syntax error", 2, "safe"),
            ("\\NAK: 3: Timeout occurred", 3, "timeout"),
            ("\\NAK: 4: Unknown error", 4, "unknown"),
        ]

        for response, expected_code, error_type in error_code_tests:
            with self.subTest(response=response, code=expected_code):
                if error_type == "safe":
                    with patch("builtins.print"):
                        result = tt._parse_response(response)
                        # Should return the message for safe errors
                        self.assertIsInstance(result, str)
                elif error_type == "timeout":
                    with self.assertRaises(TurntableTimeoutError):
                        tt._parse_response(response)
                elif error_type == "unknown":
                    with self.assertRaises(TurntableCommandError):
                        tt._parse_response(response)

    def test_parse_response_malformed_nak(self):
        """
        Test _parse_response with malformed NAK responses.
        """
        tt = Turntable.__new__(Turntable)

        malformed_naks = [
            "\\NAK: ",  # No error code
            "\\NAK: X: Invalid code",  # Non-numeric error code
            "\\NAK:",  # Missing space and content
        ]

        for response in malformed_naks:
            with self.subTest(response=response):
                with self.assertRaises(
                    (TurntableCommandError, ValueError, IndexError)
                ):
                    tt._parse_response(response)

    def test_parse_response_strips_whitespace_from_messages(self):
        """
        Test _parse_response strips whitespace from extracted messages.
        """
        tt = Turntable.__new__(Turntable)

        # Test ACK with whitespace
        ack_response = "\\ACK:   Command executed successfully   "
        result = tt._parse_response(ack_response)
        self.assertEqual(result, "Command executed successfully")

        # Test NAK safe error with whitespace
        nak_response = "\\NAK:   2: Syntax error in command   "
        with patch("builtins.print"):
            result = tt._parse_response(nak_response)
            self.assertEqual(result, "2: Syntax error in command")

    def test_rotate_default_parameters(self):
        """
        Test rotate with default speed and reverse parameters.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(
            tt, "_build_command", return_value="M390.00E"
        ) as mock_build:
            with patch.object(
                tt, "_send_command", return_value="\\ACK: Success"
            ) as mock_send:
                with patch.object(
                    tt, "_parse_response", return_value="Success"
                ) as mock_parse:
                    result = tt.rotate(90.0)

                    # Verify method calls
                    mock_build.assert_called_once_with("fast", 90.0, False)
                    mock_send.assert_called_once_with("M390.00E")
                    mock_parse.assert_called_once_with("\\ACK: Success")

                    self.assertEqual(result, "Success")

    def test_rotate_custom_parameters(self):
        """
        Test rotate with custom speed and reverse parameters.
        """
        tt = Turntable.__new__(Turntable)

        test_cases = [
            (45.5, "medium", True, "M2-45.50E"),
            (180.0, "slow", False, "M1180.00E"),
            (0.0, "fast", True, "M3-0.00E"),
            (360.0, "medium", False, "M2360.00E"),
        ]

        for angle, speed, reverse, expected_command in test_cases:
            with self.subTest(angle=angle, speed=speed, reverse=reverse):
                with patch.object(
                    tt, "_build_command", return_value=expected_command
                ) as mock_build:
                    with patch.object(
                        tt, "_send_command", return_value="\\ACK: OK"
                    ) as mock_send:
                        with patch.object(
                            tt, "_parse_response", return_value="OK"
                        ) as mock_parse:
                            result = tt.rotate(angle, speed, reverse)

                            mock_build.assert_called_once_with(
                                speed, angle, reverse
                            )
                            mock_send.assert_called_once_with(expected_command)
                            mock_parse.assert_called_once_with("\\ACK: OK")

                            self.assertEqual(result, "OK")

    def test_rotate_keyword_arguments(self):
        """
        Test rotate with keyword arguments in different orders.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(
            tt, "_build_command", return_value="M245.50E"
        ) as mock_build:
            with patch.object(
                tt, "_send_command", return_value="\\ACK: Success"
            ):
                with patch.object(
                    tt, "_parse_response", return_value="Success"
                ):
                    # Test different keyword argument orders
                    tt.rotate(angle_deg=45.5, speed="medium", reverse=True)
                    mock_build.assert_called_once_with("medium", 45.5, True)

                    mock_build.reset_mock()
                    tt.rotate(reverse=False, angle_deg=90.0, speed="slow")
                    mock_build.assert_called_once_with("slow", 90.0, False)

    def test_rotate_propagates_build_command_errors(self):
        """
        Test rotate propagates ValueError from _build_command.
        """
        tt = Turntable.__new__(Turntable)

        with patch.object(
            tt, "_build_command", side_effect=ValueError("Invalid angle")
        ):
            with self.assertRaises(ValueError) as cm:
                tt.rotate(400.0)  # Invalid angle

            self.assertIn("Invalid angle", str(cm.exception))

    def test_rotate_propagates_send_command_errors(self):
        """
        Test rotate propagates errors from _send_command.
        """
        tt = Turntable.__new__(Turntable)

        # Test TurntableConnectionError
        with patch.object(tt, "_build_command", return_value="M390.00E"):
            with patch.object(
                tt,
                "_send_command",
                side_effect=TurntableConnectionError("Port closed"),
            ):
                with self.assertRaises(TurntableConnectionError) as cm:
                    tt.rotate(90.0)

                self.assertIn("Port closed", str(cm.exception))

        # Test TurntableTimeoutError
        with patch.object(tt, "_build_command", return_value="M390.00E"):
            with patch.object(
                tt,
                "_send_command",
                side_effect=TurntableTimeoutError("Device timeout"),
            ):
                with self.assertRaises(TurntableTimeoutError) as cm:
                    tt.rotate(90.0)

                self.assertIn("Device timeout", str(cm.exception))

    def test_rotate_propagates_parse_response_errors(self):
        """
        Test rotate propagates errors from _parse_response.
        """
        tt = Turntable.__new__(Turntable)

        # Test TurntableCommandError
        with patch.object(tt, "_build_command", return_value="M390.00E"):
            with patch.object(
                tt, "_send_command", return_value="\\NAK: 4: Unknown error"
            ):
                with patch.object(
                    tt,
                    "_parse_response",
                    side_effect=TurntableCommandError("Arduino error"),
                ):
                    with self.assertRaises(TurntableCommandError) as cm:
                        tt.rotate(90.0)

                    self.assertIn("Arduino error", str(cm.exception))

    def test_rotate_method_call_sequence(self):
        """
        Test rotate calls helper methods in correct sequence.
        """
        tt = Turntable.__new__(Turntable)

        call_order = []

        def track_build_command(speed, angle, reverse):
            call_order.append(("build_command", speed, angle, reverse))
            return "M390.00E"

        def track_send_command(command):
            call_order.append(("send_command", command))
            return "\\ACK: Success"

        def track_parse_response(response):
            call_order.append(("parse_response", response))
            return "Success"

        with patch.object(
            tt, "_build_command", side_effect=track_build_command
        ):
            with patch.object(
                tt, "_send_command", side_effect=track_send_command
            ):
                with patch.object(
                    tt, "_parse_response", side_effect=track_parse_response
                ):
                    with patch("builtins.print"):  # Suppress print statements
                        tt.rotate(90.0, "fast", False)

                        expected_sequence = [
                            ("build_command", "fast", 90.0, False),
                            ("send_command", "M390.00E"),
                            ("parse_response", "\\ACK: Success"),
                        ]

                        self.assertEqual(call_order, expected_sequence)

    def test_rotate_various_responses(self):
        """
        Test rotate with various valid response types.
        """
        tt = Turntable.__new__(Turntable)

        responses = [
            ("\\ACK: Rotation complete", "Rotation complete"),
            ("\\ACK: Success", "Success"),
            ("\\NAK: 2: Syntax error", "2: Syntax error"),  # Safe error
            ("\\ACK: Command executed", "Command executed"),
        ]

        for raw_response, parsed_response in responses:
            with self.subTest(response=raw_response):
                with patch.object(
                    tt, "_build_command", return_value="M390.00E"
                ):
                    with patch.object(
                        tt, "_send_command", return_value=raw_response
                    ):
                        with patch.object(
                            tt, "_parse_response", return_value=parsed_response
                        ):
                            with patch("builtins.print"):  # Suppress prints
                                result = tt.rotate(90.0)

                                self.assertEqual(result, parsed_response)

    def test_rotate_boundary_angles(self):
        """
        Test rotate with boundary angle values.
        """
        tt = Turntable.__new__(Turntable)

        boundary_angles = [0.0, 360.0, 0.01, 359.99]

        for angle in boundary_angles:
            with self.subTest(angle=angle):
                with patch.object(
                    tt, "_build_command", return_value="M390.00E"
                ) as mock_build:
                    with patch.object(
                        tt, "_send_command", return_value="\\ACK: OK"
                    ):
                        with patch.object(
                            tt, "_parse_response", return_value="OK"
                        ):
                            with patch("builtins.print"):
                                result = tt.rotate(angle)

                                mock_build.assert_called_once_with(
                                    "fast", angle, False
                                )
                                self.assertEqual(result, "OK")

    def test_rotate_all_speed_options(self):
        """
        Test rotate with all valid speed options.
        """
        tt = Turntable.__new__(Turntable)

        speeds = ["slow", "medium", "fast"]

        for speed in speeds:
            with self.subTest(speed=speed):
                with patch.object(
                    tt, "_build_command", return_value="M390.00E"
                ) as mock_build:
                    with patch.object(
                        tt, "_send_command", return_value="\\ACK: OK"
                    ):
                        with patch.object(
                            tt, "_parse_response", return_value="OK"
                        ):
                            with patch("builtins.print"):
                                tt.rotate(90.0, speed)

                                mock_build.assert_called_once_with(
                                    speed, 90.0, False
                                )

    def test_rotate_both_directions(self):
        """
        Test rotate with both forward and reverse directions.
        """
        tt = Turntable.__new__(Turntable)

        directions = [
            False,
            True,
        ]  # False = clockwise, True = counterclockwise

        for reverse in directions:
            with self.subTest(reverse=reverse):
                with patch.object(
                    tt, "_build_command", return_value="M390.00E"
                ) as mock_build:
                    with patch.object(
                        tt, "_send_command", return_value="\\ACK: OK"
                    ):
                        with patch.object(
                            tt, "_parse_response", return_value="OK"
                        ):
                            with patch("builtins.print"):
                                tt.rotate(90.0, reverse=reverse)

                                mock_build.assert_called_once_with(
                                    "fast", 90.0, reverse
                                )

    def test_rotate_no_side_effects(self):
        """
        Test rotate doesn't modify input parameters.
        """
        tt = Turntable.__new__(Turntable)

        original_angle = 90.0
        original_speed = "medium"
        original_reverse = True

        angle_deg = original_angle
        speed = original_speed
        reverse = original_reverse

        with patch.object(tt, "_build_command", return_value="M390.00E"):
            with patch.object(tt, "_send_command", return_value="\\ACK: OK"):
                with patch.object(tt, "_parse_response", return_value="OK"):
                    with patch("builtins.print"):
                        tt.rotate(angle_deg, speed, reverse)

        # Verify inputs weren't modified
        self.assertEqual(angle_deg, original_angle)
        self.assertEqual(speed, original_speed)
        self.assertEqual(reverse, original_reverse)

    def test_exception_classes_inheritance(self):
        """Test that exception classes inherit correctly."""
        self.assertTrue(issubclass(TurntableConnectionError, TurntableError))
        self.assertTrue(issubclass(TurntableCommandError, TurntableError))
        self.assertTrue(issubclass(TurntableTimeoutError, TurntableError))

    def test_constants_are_defined(self):
        """Test that all required class constants exist."""
        tt = Turntable.__new__(Turntable)
        required_constants = [
            "START_BYTE",
            "END_BYTE",
            "SPEED_MAPPING",
            "MIN_ANGLE_DEG",
            "MAX_ANGLE_DEG",
            "ANGLE_PRECISION",
        ]
        for constant in required_constants:
            self.assertTrue(hasattr(tt, constant))


unittest.main()
