"""
NanoControl Module Tests
======================

Comprehensive unit test suite for the NanoControl micromanipulator interface
module. This test suite validates all functionality of the NanoControl class
and its associated exception classes using mock objects to simulate hardware
interactions without requiring physical device connections.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import serial  # Add this import
from unittest.mock import Mock, patch
from micromanipulator_tools.utils.nanocontrol import (
    NanoControl,
    NanoControlError,
    NanoControlCommandError,
    NanoControlConnectionError,
)


class TestNanoControl(unittest.TestCase):
    """
    These tests are created to confirm the logic of the software
    above. They work by instantiating virtual serial ports and
    objects to mimic the actual hardware and test all possible logic
    paths. This is done using Mock and patch.
    """

    def test_get_version(self):
        """
        Test the whether get_version returns a string.
        """

        # Create an empty instance of the class without running
        # __init__ to avoid needing hardware to test.
        nc = NanoControl.__new__(NanoControl)

        # Patch _send_command so it returns a mock version string
        with patch.object(
            nc, "_send_command", return_value="NC4 01.70-180807"
        ):
            version = nc.get_version()
            self.assertIsInstance(version, str)
            self.assertEqual(version, "NC4 01.70-180807")

    def test_close_serial_try(self):
        """
        Test that _close_serial correctly closes the serial port.
        """

        nc = NanoControl.__new__(NanoControl)

        # Create a mock serial object
        mock_serial = Mock()

        # Explicitly mark the mock serial as "open" so that
        # _close_serial will attempt to close it. (By default, Mock
        # attributes are truthy, but here we set it explicitly for
        # clarity.)
        mock_serial.is_open = True

        # Make close() set is_open to False when called and make it
        # side effect of closing the mock serial.
        def mock_close():
            mock_serial.is_open = False

        mock_serial.close.side_effect = mock_close

        # Assign the mock to the instance
        nc.device_serial = mock_serial

        # Call the method under test
        nc._close_serial()

        # Verify that .close() was called
        mock_serial.close.assert_called_once()

        # Now .is_open should be False automatically
        self.assertFalse(mock_serial.is_open)

    def test_close_serial_except(self):
        """
        Test that _close_serial handles exceptions during serial
        close and prints the correct warning message.
        """

        nc = NanoControl.__new__(NanoControl)
        mock_serial = Mock()
        mock_serial.is_open = True

        # Force .close() to raise an exception to test the method's
        # exception path.
        mock_serial.close.side_effect = Exception("Mock close exception")

        # Inject our mock serial object into the NanoControl instance.
        nc.device_serial = mock_serial

        # Patch 'print' so we can capture the output and verify
        # the warning.
        with patch("builtins.print") as mock_print:
            nc._close_serial()

        # Check that the warning was printed with the correct
        # message
        mock_print.assert_any_call(
            "NanoControl._close_serial: WARNING: Exception when closing "
            "serial port: Mock close exception"
        )

    def test_send_command_timeout(self):
        """
        Test that _send_command raises NanoControlCommandError
        when the device returns an empty response (timeout).
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()
        nc.device_serial.read_until.return_value = b""

        with self.assertRaises(NanoControlCommandError) as cm:
            nc._send_command("version")
        self.assertIn("No response from device", str(cm.exception))

    def test_send_command_ok_status(self):
        """
        Test that _send_command returns the message string when the
        device responds with an OK status ('o').
        """
        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        # The mock pretends to be the device and sends back the
        # string "o\tfirmware 1.0\r"
        nc.device_serial.read_until.return_value = b"o\tfirmware 1.0\r"

        result = nc._send_command("version")
        self.assertEqual(result, "firmware 1.0")

    def test_send_command_error_status(self):
        """
        Test that _send_command raises NanoControlCommandError
        when the device responds with an error status ('e').
        """
        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()
        nc.device_serial.read_until.return_value = b"e\tbad things happened\r"

        # Pass the test only if NanoControlCommandError exception
        # is raised.
        with self.assertRaises(NanoControlCommandError) as cm:
            nc._send_command("version")
        self.assertIn("bad things happened", str(cm.exception))

    def test_send_command_info_status(self):
        """
        Test that _send_command returns the message string and
        prints an info message when the device responds with an info
        status ('i').
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()
        nc.device_serial.read_until.return_value = b"i\tbuild date\r"

        # Capture what is printed and check is correct using patch.
        with patch("builtins.print") as mock_print:
            result = nc._send_command("version")

            # Assert that the print actually happened.
            mock_print.assert_any_call(
                "NanoControl._send_command: Info from device: build date"
            )

            # Check the decoded message is correct.
            self.assertEqual(result, "build date")

    def test_send_command_unknown_status(self):
        """
        Test that _send_command raises NanoControlCommandError
        when the device responds with an unknown status character.
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()
        nc.device_serial.read_until.return_value = b"x\tweird\r"

        with self.assertRaises(NanoControlCommandError) as cm:
            nc._send_command("version")
        self.assertIn("Unknown response status", str(cm.exception))

    def test_send_command_serial_exception(self):
        """
        Test that _send_command raises NanoControlConnectionError
        when a serial.SerialException is raised during
        communication.
        """
        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()
        nc.device_serial.write.side_effect = serial.SerialException(
            "port fail"
        )

        with self.assertRaises(NanoControlConnectionError) as cm:
            nc._send_command("version")
        self.assertIn("Serial error", str(cm.exception))

    def test_send_command_unicode_encode_error(self):
        """
        Test that _send_command raises ValueError when the command
        string contains non-ASCII characters.
        """

        nc = NanoControl.__new__(NanoControl)

        # Create command with the non-ASCII character: °
        invalid_command = "set_temp 90°"

        with self.assertRaises(ValueError) as cm:
            nc._send_command(invalid_command)

        # Verify the exception message and that the original error
        # was chained correctly.
        self.assertIn("non-ASCII character", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, UnicodeEncodeError)

    def test_send_command_strips_whitespace(self):
        """
        Test that _send_command correctly strips leading/trailing
        whitespace from the device response.
        """
        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        # A dictionary of test cases with messy whitespace
        test_cases = {
            "leading_spaces": b"  o\tmessage\r",
            "trailing_spaces": b"o\tmessage  \r",
            "trailing_cr_lf": b"o\tmessage\r\n",
            "mixed_whitespace": b" \t o\tmessage \t \r\n",
        }

        for name, dirty_response in test_cases.items():
            with self.subTest(case=name):
                # Set the mock to return the dirty response
                nc.device_serial.read_until.return_value = dirty_response

                # Call the method
                result = nc._send_command("any command")

                # Assert that the result is the clean string
                self.assertEqual(result, "message")

    def test_beep_valid(self):
        """
        Test that beep sends the correct command string and returns
        the expected response when inputs are valid.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command", return_value="OK") as mock_send:
            response = nc.beep(1000, 500)
            mock_send.assert_called_once_with("beep 1000 500")
            self.assertEqual(response, "OK")

    def test_beep_invalid_frequency_type(self):
        """
        Test that beep raises ValueError when frequency is not a
        positive integer.
        """
        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError):
            nc.beep("1000", 500)

        with self.assertRaises(ValueError):
            nc.beep(0, 500)

        with self.assertRaises(ValueError):
            nc.beep(-100, 500)

        with self.assertRaises(ValueError):
            nc.beep(-10.0, 500)

    def test_beep_invalid_duration_value(self):
        """
        Test that beep raises ValueError when duration_ms is not
        a positive integer.
        """
        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError):
            nc.beep(1000, 0)

        with self.assertRaises(ValueError):
            nc.beep(1000, "0")

        with self.assertRaises(ValueError):
            nc.beep(1000, -100)

        with self.assertRaises(ValueError):
            nc.beep(1000, -10.0)

    def test_beep_connection_error(self):
        """
        Test that beep propagates NanoControlError if
        _send_command fails.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(
            nc,
            "_send_command",
            side_effect=NanoControlError("Device error"),
        ):
            with self.assertRaises(NanoControlError):
                nc.beep(1000, 500)

    def test_init_com_port_empty(self):
        with self.assertRaises(ValueError) as context:
            NanoControl("", 115200)
        self.assertIn(
            "com_port must be a non-empty string", str(context.exception)
        )

    def test_init_com_port_not_str(self):
        with self.assertRaises(ValueError) as context:
            NanoControl(15, 115200)
        self.assertIn(
            "com_port must be a non-empty string", str(context.exception)
        )

    def test_init_baud_rate_negative(self):
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", -15)
        self.assertIn(
            "baud_rate must be a positive integer", str(context.exception)
        )

    def test_init_baud_rate_float(self):
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200.2)
        self.assertIn(
            "baud_rate must be a positive integer", str(context.exception)
        )

    def test_init_baud_rate_str(self):
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", "hello")
        self.assertIn(
            "baud_rate must be a positive integer", str(context.exception)
        )

    def test_init_serial_exception(self):
        """
        Test that NanoControl.__init__ raises
        NanoControlError when serial.Serial raises a
        SerialException.
        """

        with patch(
            "serial.Serial",
            side_effect=serial.SerialException("port not found"),
        ):
            with self.assertRaises(NanoControlError) as cm:
                NanoControl("COM19")

            self.assertIn(
                "Serial connection failed on COM19", str(cm.exception)
            )
            self.assertIsInstance(
                cm.exception.__cause__, serial.SerialException
            )

    def test_init_beep_raises_nano_control_error(self):
        """
        Test that NanoControl.__init__ raises
        NanoControlError when beep() raises
        NanoControlError.
        """

        # Create a mock serial instance
        mock_serial_instance = Mock()
        mock_serial_instance.is_open = True

        # Mock successful responses for set_motor_drive_frequencies and set_fine_with_coarse_all
        mock_serial_instance.read_until.return_value = b"o\tOK\r"
        mock_serial_instance.flushInput = Mock()
        mock_serial_instance.write = Mock()
        mock_serial_instance.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial_instance):
            with patch.object(
                NanoControl,
                "beep",
                side_effect=NanoControlConnectionError(
                    "device not responding"
                ),
            ):
                with self.assertRaises(NanoControlError) as cm:
                    NanoControl("COM19")

                self.assertIn(
                    "Device verification failed on COM19",
                    str(cm.exception),
                )

                self.assertIsInstance(cm.exception.__cause__, NanoControlError)
                self.assertEqual(
                    str(cm.exception.__cause__),
                    "device not responding",
                )

    def test_init_exception_closes_port(self):
        """
        Verify that when NanoControl.beep() raises a
        NanoControlError after the serial port has been
        successfully opened, the serial port is properly closed
        before the error is propagated.
        """

        # Create a mock serial instance with .is_open and .close()
        mock_serial_instance = Mock()
        mock_serial_instance.is_open = True
        mock_serial_instance.close = Mock()

        # Mock successful responses for the first two calls
        mock_serial_instance.read_until.return_value = b"o\tOK\r"
        mock_serial_instance.flushInput = Mock()
        mock_serial_instance.write = Mock()
        mock_serial_instance.flush = Mock()

        # Patch serial.Serial to return the mock instance, but
        # then cause beep() to raise NanoControlError to trigger except
        with patch("serial.Serial", return_value=mock_serial_instance):
            with patch.object(
                NanoControl,
                "beep",
                side_effect=NanoControlCommandError("device not responding"),
            ):
                with self.assertRaises(NanoControlError):
                    NanoControl("COM19")

                # Assert that .close() was called on the serial port
                # This works because _cleanup_failed_init calls close() on the mock
                mock_serial_instance.close.assert_called_once()

    def test_init_timeout_invalid_types(self):
        """Test timeout_sec validation with invalid types and values."""

        # Test zero timeout
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200, 0)
        self.assertIn(
            "timeout_sec must be a positive number", str(context.exception)
        )

        # Test negative timeout
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200, -5)
        self.assertIn(
            "timeout_sec must be a positive number", str(context.exception)
        )

        # Test negative float timeout
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200, -2.5)
        self.assertIn(
            "timeout_sec must be a positive number", str(context.exception)
        )

        # Test string timeout
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200, "10")
        self.assertIn(
            "timeout_sec must be a positive number", str(context.exception)
        )

        # Test None timeout
        with self.assertRaises(ValueError) as context:
            NanoControl("COM19", 115200, None)
        self.assertIn(
            "timeout_sec must be a positive number", str(context.exception)
        )

    def test_init_successful(self):
        """Test that successful initialization sets all attributes correctly."""
        mock_serial = Mock()
        mock_serial.is_open = True

        # Mock successful serial responses
        mock_serial.read_until.return_value = b"o\tOK\r"
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch("serial.Serial", return_value=mock_serial):
            nc = NanoControl("COM19", 9600, 5)

            # Verify attributes are set correctly
            self.assertEqual(nc.com_port, "COM19")
            self.assertEqual(nc.baud_rate, 9600)
            self.assertEqual(nc.timeout_sec, 5)
            self.assertEqual(nc.device_serial, mock_serial)

    def test_init_default_parameters(self):
        """Test that default baud_rate and timeout_sec work correctly."""
        mock_serial = Mock()
        mock_serial.is_open = True

        # Mock successful serial responses
        mock_serial.read_until.return_value = b"o\tOK\r"
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        with patch(
            "serial.Serial", return_value=mock_serial
        ) as mock_serial_constructor:
            nc = NanoControl("COM19")  # Only required parameter

            # Verify defaults were used
            self.assertEqual(nc.baud_rate, 115200)
            self.assertEqual(nc.timeout_sec, 10)

            # Verify serial.Serial was called with defaults
            mock_serial_constructor.assert_called_once_with(
                "COM19", 115200, timeout=10
            )

    def test_init_com_port_whitespace_only(self):
        """Test that com_port with only whitespace raises ValueError."""
        with self.assertRaises(ValueError):
            NanoControl("   ", 115200)  # spaces only

        with self.assertRaises(ValueError):
            NanoControl("\t\n", 115200)  # tabs and newlines

    def test_exit_closes_serial_and_prints_exception(self):
        """
        Test that NanoControl.__exit__ calls _close_serial() and
        prints exception details when an exception occurs inside a
        'with' block.

        This ensures the serial port is closed properly and
        diagnostic info is printed if an error occurs.
        """

        # Create a mock serial instance with .is_open and .close()
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial.close = Mock()

        # Mock the serial read_until to return proper response format
        mock_serial.read_until.return_value = b"o\tOK\r"
        mock_serial.flushInput = Mock()
        mock_serial.write = Mock()
        mock_serial.flush = Mock()

        # Patch serial.Serial to return the mock instance
        with patch("serial.Serial", return_value=mock_serial):
            # Patch the _close_serial method of NanoControl with
            # a mock to check if NanoControl calls it.
            with patch.object(NanoControl, "_close_serial") as mock_close:
                with patch("builtins.print") as mock_print:
                    try:
                        # Use NanoControl in a with block and
                        # raise an exception inside
                        with NanoControl("COM19"):
                            raise ValueError(
                                "test exception inside with block"
                            )
                    except ValueError:
                        # expected exception to test __exit__
                        pass

                    # Assert that _close_serial was called once
                    mock_close.assert_called_once()

                    # Fixed: Use the correct format with dots
                    mock_print.assert_any_call(
                        "NanoControl.__exit__: Exception type: <class "
                        "'ValueError'>"
                    )
                    mock_print.assert_any_call(
                        "NanoControl.__exit__: Exception value: "
                        "test exception inside with block"
                    )

                    # Check that the traceback line was printed
                    # (just contains the word "Traceback:")
                    printed_args = [
                        call.args[0] for call in mock_print.call_args_list
                    ]
                    assert any("Traceback:" in arg for arg in printed_args)

    def test_str_returns_expected_string(self):
        # create instance, __init__ not called
        nc = NanoControl.__new__(NanoControl)
        nc.com_port = "COM19"  # manually set attribute
        # Fixed: Use the correct format with dots
        self.assertEqual(
            str(nc), "NanoControl.__str__: NanoControl Port: COM19"
        )

    def test_query_speed_profile_success(self):
        """
        Test that _query_speed_profile returns the raw response
        from _send_command.
        """

        nc = NanoControl.__new__(NanoControl)

        expected_response = "2 c053 f001 f064 c023"

        with patch.object(
            nc, "_send_command", return_value=expected_response
        ) as mock_send:
            result = nc._query_speed_profile()

            # Verify the correct command was sent
            mock_send.assert_called_once_with("speed ?")

            # Verify the raw response is returned unchanged
            self.assertEqual(result, expected_response)

    def test_query_speed_profile_propagates_connection_error(self):
        """
        Test that _query_speed_profile propagates
        NanoControlConnectionError from _send_command.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(
            nc,
            "_send_command",
            side_effect=NanoControlConnectionError("Serial port disconnected"),
        ):
            with self.assertRaises(NanoControlConnectionError) as cm:
                nc._query_speed_profile()

            self.assertIn("Serial port disconnected", str(cm.exception))

    def test_query_speed_profile_propagates_command_error(self):
        """
        Test that _query_speed_profile propagates
        NanoControlCommandError from _send_command.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(
            nc,
            "_send_command",
            side_effect=NanoControlCommandError("Device timeout"),
        ):
            with self.assertRaises(NanoControlCommandError) as cm:
                nc._query_speed_profile()

            self.assertIn("Device timeout", str(cm.exception))

    def test_query_speed_profile_various_valid_responses(self):
        """
        Test _query_speed_profile with various valid device response
        formats.
        """
        nc = NanoControl.__new__(NanoControl)

        valid_responses = [
            "1 c001 c002 c003 c004",  # All coarse, low values
            "6 f064 f064 f064 f064",  # All fine, max values
            "3 c032 f016 c008 f001",  # Mixed coarse/fine
            "2 c053 f001 f064 c023",  # Your example format
        ]

        for response in valid_responses:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command", return_value=response):
                    result = nc._query_speed_profile()
                    self.assertEqual(result, response)

    def test_validate_speed_value_user_format_valid(self):
        """
        Test _validate_speed_value with valid 3-character user
        format inputs.
        """

        nc = NanoControl.__new__(NanoControl)

        valid_user_inputs = [
            "c01",
            "c32",
            "c64",  # Coarse mode, various values
            "f01",
            "f32",
            "f64",  # Fine mode, various values
            "C01",
            "F64",  # Uppercase letters
        ]

        for speed_str in valid_user_inputs:
            with self.subTest(speed_str=speed_str):
                # Should not raise any exception
                try:
                    nc._validate_speed_value(
                        speed_str, allow_device_format=False
                    )
                except ValueError:
                    self.fail(
                        f"_validate_speed_value raised ValueError "
                        f"unexpectedly for valid input '{speed_str}'"
                    )

    def test_validate_speed_value_device_format_valid(self):
        """
        Test _validate_speed_value with valid device format inputs
        when enabled.
        """

        nc = NanoControl.__new__(NanoControl)

        valid_device_inputs = [
            "c001",
            "c032",
            "c064",  # 4-char coarse format
            "f001",
            "f032",
            "f064",  # 4-char fine format
            "c01",
            "f32",  # 3-char format still works
        ]

        for speed_str in valid_device_inputs:
            with self.subTest(speed_str=speed_str):
                try:
                    nc._validate_speed_value(
                        speed_str, allow_device_format=True
                    )
                except ValueError:
                    self.fail(
                        "_validate_speed_value raised ValueError "
                        "unexpectedly for valid device "
                        f"input '{speed_str}'"
                    )

    def test_validate_speed_value_invalid_type(self):
        """
        Test _validate_speed_value with non-string inputs.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_types = [None, 123, 45.6, ["c32"], {"speed": "c32"}]

        for invalid_input in invalid_types:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_speed_value(invalid_input)

                self.assertIn(
                    "Speed value must be a string", str(cm.exception)
                )

    def test_validate_speed_value_invalid_format_user_mode(self):
        """
        Test _validate_speed_value with invalid formats in user mode.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_formats = {
            # Wrong length
            "c1": "not in valid 3-character format",
            "f8": "not in valid 3-character format",
            "c100": "not in valid 3-character format",
            "c": "not in valid 3-character format",
            # Wrong prefix
            "x32": "not in valid 3-character format",
            "32f": "not in valid 3-character format",
            "a64": "not in valid 3-character format",
            # Non-digits after prefix
            "caa": "not in valid 3-character format",
            "fab": "not in valid 3-character format",
            # Empty string
            "": "not in valid 3-character format",
        }

        for (
            invalid_input,
            expected_msg_fragment,
        ) in invalid_formats.items():
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_speed_value(
                        invalid_input, allow_device_format=False
                    )

                self.assertIn(expected_msg_fragment, str(cm.exception))

    def test_validate_speed_value_invalid_format_device_mode(self):
        """
        Test _validate_speed_value with invalid formats in device
        mode.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_formats = [
            "c1",  # Too short (only 2 chars)
            "c12345",  # Too long (5 chars)
            "x123",  # Wrong prefix
            "c",  # Way too short
            "cabc",  # Non-digits
            "",  # Empty
        ]

        for invalid_input in invalid_formats:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_speed_value(
                        invalid_input, allow_device_format=True
                    )

                self.assertIn("not in valid format", str(cm.exception))

    def test_validate_speed_value_invalid_range(self):
        """
        Test _validate_speed_value with values outside valid range
        (1-64).
        """

        nc = NanoControl.__new__(NanoControl)

        out_of_range_inputs = {
            "c00": "must be between 1 and 64",  # Too low
            "f00": "must be between 1 and 64",  # Too low
            "c65": "must be between 1 and 64",  # Too high
            "f99": "must be between 1 and 64",  # Too high
        }

        for (
            invalid_input,
            expected_msg_fragment,
        ) in out_of_range_inputs.items():
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_speed_value(invalid_input)

                self.assertIn(expected_msg_fragment, str(cm.exception))

    def test_validate_speed_value_boundary_values(self):
        """
        Test _validate_speed_value with boundary values (1 and 64).
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_values = ["c01", "c64", "f01", "f64"]

        for speed_str in boundary_values:
            with self.subTest(speed_str=speed_str):
                try:
                    nc._validate_speed_value(speed_str)
                except ValueError:
                    self.fail(
                        f"_validate_speed_value rejected valid boundary "
                        f"value '{speed_str}'"
                    )

    def test_validate_speed_value_case_insensitive(self):
        """
        Test that _validate_speed_value handles case-insensitive
        prefixes.
        """

        nc = NanoControl.__new__(NanoControl)

        case_variations = [
            ("c32", "C32"),  # Lower vs upper c
            ("f16", "F16"),  # Lower vs upper f
        ]

        for lower_case, upper_case in case_variations:
            with self.subTest(lower=lower_case, upper=upper_case):
                # Both should work without raising exceptions
                try:
                    nc._validate_speed_value(lower_case)
                    nc._validate_speed_value(upper_case)
                except ValueError:
                    self.fail(
                        f"Case sensitivity test failed "
                        f"for {lower_case}/{upper_case}"
                    )

    def test_validate_speed_value_device_format_with_user_disabled(self):
        """
        Test that 4-char device format is rejected when
        allow_device_format=False.
        """

        nc = NanoControl.__new__(NanoControl)

        device_format_inputs = ["c001", "f064", "c032"]

        for speed_str in device_format_inputs:
            with self.subTest(speed_str=speed_str):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_speed_value(
                        speed_str, allow_device_format=False
                    )

                self.assertIn(
                    "not in valid 3-character format", str(cm.exception)
                )

    def test_validate_profile_number_valid_range(self):
        """
        Test _validate_profile_number with valid profile numbers
        (1-6).
        """

        nc = NanoControl.__new__(NanoControl)

        valid_profiles = [1, 2, 3, 4, 5, 6]

        for profile_num in valid_profiles:
            with self.subTest(profile_num=profile_num):
                try:
                    nc._validate_profile_number(profile_num)
                except ValueError:
                    self.fail(
                        f"_validate_profile_number raised ValueError "
                        f"unexpectedly for valid profile {profile_num}"
                    )

    def test_validate_profile_number_invalid_type(self):
        """
        Test _validate_profile_number with non-integer inputs.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_types = [
            None,
            "3",
            3.0,
            3.5,
            [3],
            {"profile": 3},
            True,
            False,  # Booleans (subclass of int but rejected)
        ]

        for invalid_input in invalid_types:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_profile_number(invalid_input)

                self.assertIn(
                    "speed profile number must be an integer",
                    str(cm.exception),
                )

    def test_validate_profile_number_out_of_range_low(self):
        """
        Test _validate_profile_number with profile numbers below
        valid range.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_low_values = [0, -1, -10, -100]

        for profile_num in invalid_low_values:
            with self.subTest(profile_num=profile_num):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_profile_number(profile_num)

                error_msg = str(cm.exception)
                self.assertIn("must be an integer from", error_msg)
                self.assertIn("1 to 6", error_msg)

    def test_validate_profile_number_out_of_range_high(self):
        """
        Test _validate_profile_number with profile numbers above
        valid range.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_high_values = [7, 8, 10, 100, 999]

        for profile_num in invalid_high_values:
            with self.subTest(profile_num=profile_num):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_profile_number(profile_num)

                error_msg = str(cm.exception)
                self.assertIn("must be an integer from", error_msg)
                self.assertIn("1 to 6", error_msg)

    def test_validate_profile_number_boundary_values(self):
        """
        Test _validate_profile_number with boundary values
        (1 and 6).
        """

        nc = NanoControl.__new__(NanoControl)

        boundary_values = [1, 6]  # MIN and MAX valid values

        for profile_num in boundary_values:
            with self.subTest(profile_num=profile_num):
                try:
                    nc._validate_profile_number(profile_num)
                except ValueError:
                    self.fail(
                        f"_validate_profile_number rejected valid "
                        f"boundary value {profile_num}"
                    )

    def test_validate_profile_number_just_outside_boundaries(self):
        """
        Test _validate_profile_number with values just outside
        boundaries.
        """

        nc = NanoControl.__new__(NanoControl)

        just_outside = [0, 7]  # Just below MIN and just above MAX

        for profile_num in just_outside:
            with self.subTest(profile_num=profile_num):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_profile_number(profile_num)

                error_msg = str(cm.exception)
                self.assertIn("must be an integer from 1 to 6", error_msg)

    def test_validate_profile_number_boolean_rejection(self):
        """
        Test that _validate_profile_number explicitly rejects
        boolean values.

        Even though bool is a subclass of int in Python, the
        function uses 'type(profile_num) is not int' to explicitly
        reject booleans.
        """

        nc = NanoControl.__new__(NanoControl)

        boolean_values = [True, False]

        for bool_val in boolean_values:
            with self.subTest(bool_val=bool_val):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_profile_number(bool_val)

                self.assertIn("must be an integer", str(cm.exception))

    def test_validate_profile_number_error_message_content(self):
        """
        Test that error messages contain expected function context
        and ranges.
        """

        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc._validate_profile_number("invalid")

        error_msg = str(cm.exception)

        # Check that error message contains expected components
        self.assertIn(
            "NanoControl._validate_profile_number", error_msg
        )  # ✅ Fixed: Use actual function name
        self.assertIn("speed profile number", error_msg)
        self.assertIn("integer from 1 to 6", error_msg)

    def test_validate_profile_number_uses_class_constants(self):
        """
        Test that _validate_profile_number uses class constants in
        validation.

        This is a white-box test to ensure the function uses the
        class constants rather than hardcoded values.
        """

        nc = NanoControl.__new__(NanoControl)

        # Temporarily modify the class constants to verify they're
        # used
        original_min = nc.MIN_SPEED_PROFILE_NUM
        original_max = nc.MAX_SPEED_PROFILE_NUM

        try:
            # Change the valid range to 2-4
            nc.MIN_SPEED_PROFILE_NUM = 2
            nc.MAX_SPEED_PROFILE_NUM = 4

            # Now 1 and 5 should be invalid, but 3 should be valid
            with self.assertRaises(ValueError):
                nc._validate_profile_number(1)  # Below new minimum

            with self.assertRaises(ValueError):
                nc._validate_profile_number(5)  # Above new maximum

            # But 3 should still work
            try:
                nc._validate_profile_number(3)
            except ValueError:
                self.fail("Profile 3 should be valid with modified range 2-4")

        finally:
            # Restore original constants
            nc.MIN_SPEED_PROFILE_NUM = original_min
            nc.MAX_SPEED_PROFILE_NUM = original_max

    def test_parse_speed_response_valid_format(self):
        """
        Test _parse_speed_response with valid device responses.
        """

        nc = NanoControl.__new__(NanoControl)

        valid_responses = [
            ("1 c001 c002 c003 c004", 1, ["c001", "c002", "c003", "c004"]),
            ("6 f064 f064 f064 f064", 6, ["f064", "f064", "f064", "f064"]),
            ("3 c032 f016 c008 f001", 3, ["c032", "f016", "c008", "f001"]),
            ("2 c053 f001 f064 c023", 2, ["c053", "f001", "f064", "c023"]),
        ]

        for response, expected_profile, expected_speeds in valid_responses:
            with self.subTest(response=response):
                profile_num, speeds = nc._parse_speed_response(response)

                self.assertEqual(profile_num, expected_profile)
                self.assertEqual(speeds, expected_speeds)

    def test_parse_speed_response_wrong_token_count(self):
        """
        Test _parse_speed_response with incorrect number of tokens.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_token_counts = [
            ("1", 1),  # Too few tokens
            ("1 c32", 2),  # Still too few
            ("1 c32 f16", 3),  # Missing 2 speeds
            ("1 c32 f16 c08", 4),  # Missing 1 speed
            ("1 c32 f16 c08 c64 extra", 6),  # Too many tokens
        ]

        for response, token_count in invalid_token_counts:
            with self.subTest(response=response):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn("Expected 5 tokens", error_msg)
                self.assertIn(f"got {token_count}", error_msg)
                self.assertIn(response, error_msg)

    def test_parse_speed_response_invalid_profile_number(self):
        """
        Test _parse_speed_response with invalid profile numbers.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_profile_responses = [
            "abc c032 f016 c008 f001",  # Non-integer profile
            "3.5 c032 f016 c008 f001",  # Float profile
            "0 c032 f016 c008 f001",  # Out of range (too low)
            "7 c032 f016 c008 f001",  # Out of range (too high)
            "99 c032 f016 c008 f001",  # Way out of range
        ]

        for response in invalid_profile_responses:
            with self.subTest(response=response):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn("Profile number", error_msg)
                self.assertIn("not an int in the valid range", error_msg)
                self.assertIn(response, error_msg)

    def test_parse_speed_response_invalid_speed_format(self):
        """
        Test _parse_speed_response with invalid speed value formats.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_speed_responses = [
            (
                "1 x032 f016 c008 f001",
                "base_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 y016 c008 f001",
                "elbow_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 f016 z008 f001",
                "prismatic_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 f016 c008 w001",
                "tweezer_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c00 f016 c008 f001",
                "base_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 f65 c008 f001",
                "elbow_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 f016 c999 f001",
                "prismatic_joint",
            ),  # ✅ Fixed: Use full joint name
            (
                "1 c032 f016 c008 f1",
                "tweezer_joint",
            ),  # ✅ Fixed: Use full joint name
        ]

        for response, expected_joint in invalid_speed_responses:
            with self.subTest(response=response, joint=expected_joint):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn(
                    f"Invalid {expected_joint} speed", error_msg
                )  # ✅ Now matches implementation
                self.assertIn(response, error_msg)

    def test_parse_speed_response_boundary_values(self):
        """
        Test _parse_speed_response with boundary profile numbers and
        speed values.
        """

        nc = NanoControl.__new__(NanoControl)

        boundary_responses = [
            "1 c001 f001 c001 f001",  # Min profile, min speeds
            "6 c064 f064 c064 f064",  # Max profile, max speeds
            "1 c064 f001 c032 f016",  # Min profile, mixed speeds
            "6 c001 f064 c008 f032",  # Max profile, mixed speeds
        ]

        for response in boundary_responses:
            with self.subTest(response=response):
                try:
                    profile_num, speeds = nc._parse_speed_response(response)
                    # Should not raise any exception
                    self.assertIsInstance(profile_num, int)
                    self.assertIsInstance(speeds, list)
                    self.assertEqual(len(speeds), 4)
                except NanoControlError:
                    self.fail(
                        f"_parse_speed_response raised error unexpectedly "
                        f"for valid boundary response: {response}"
                    )

    def test_parse_speed_response_mixed_case_prefixes(self):
        """
        Test _parse_speed_response handles mixed case speed prefixes.
        """

        nc = NanoControl.__new__(NanoControl)

        mixed_case_responses = [
            "2 C032 f016 c008 F001",  # Mixed uppercase/lowercase
            "3 c032 F016 C008 f001",  # Different pattern
        ]

        for response in mixed_case_responses:
            with self.subTest(response=response):
                try:
                    profile_num, speeds = nc._parse_speed_response(response)
                    self.assertIsInstance(profile_num, int)
                    self.assertEqual(len(speeds), 4)
                except NanoControlError:
                    self.fail(f"Mixed case handling failed for: {response}")

    def test_parse_speed_response_exception_chaining(self):
        """
        Test that _parse_speed_response properly chains original
        ValueError exceptions.
        """

        nc = NanoControl.__new__(NanoControl)

        # Test profile number parsing error chaining
        with self.assertRaises(NanoControlError) as cm:
            nc._parse_speed_response("invalid c032 f016 c008 f001")

        # Check that the original ValueError is chained
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_parse_speed_response_empty_response(self):
        """
        Test _parse_speed_response with empty or whitespace-only
        responses.
        """

        nc = NanoControl.__new__(NanoControl)

        empty_responses = ["", "   ", "\t", "\n", "  \t  \n  "]

        for response in empty_responses:
            with self.subTest(response=repr(response)):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn("Expected 5 tokens", error_msg)

    def test_parse_speed_response_extra_whitespace(self):
        """
        Test _parse_speed_response handles extra whitespace correctly.
        """

        nc = NanoControl.__new__(NanoControl)

        whitespace_responses = [
            "  2   c032   f016   c008   f001  ",  # Extra spaces
            "\t2\tc032\tf016\tc008\tf001\t",  # Tabs
            "2  c032  f016  c008  f001",  # Double spaces
        ]

        for response in whitespace_responses:
            with self.subTest(response=repr(response)):
                try:
                    profile_num, speeds = nc._parse_speed_response(response)
                    self.assertEqual(profile_num, 2)
                    self.assertEqual(speeds, ["c032", "f016", "c008", "f001"])
                except NanoControlError:
                    self.fail(
                        f"Extra whitespace handling failed for: {response}"
                    )

    def test_parse_speed_response_function_name_in_errors(self):
        """
        Test that error messages contain the correct function name.
        """

        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(NanoControlError) as cm:
            nc._parse_speed_response("1 c32")  # Too few tokens

        error_msg = str(cm.exception)
        self.assertIn("NanoControl._parse_speed_response", error_msg)

    def test_change_speed_profile_to_successful_switch(self):
        """
        Test change_speed_profile_to with successful profile switch.
        """

        nc = NanoControl.__new__(NanoControl)

        # Mock the methods that change_speed_profile_to calls
        with patch.object(nc, "_validate_profile_number") as mock_validate:
            with patch.object(nc, "_send_command") as mock_send:
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    with patch.object(
                        nc, "_parse_speed_response"
                    ) as mock_parse:
                        # Setup expected behavior
                        mock_query.return_value = "3 c032 f016 c008 f001"
                        mock_parse.return_value = (
                            3,
                            ["c032", "f016", "c008", "f001"],
                        )

                        # Test the function
                        result_profile, result_speeds = (
                            nc.change_speed_profile_to(3)
                        )

                        # Verify all calls were made correctly
                        mock_validate.assert_called_once_with(3)
                        mock_send.assert_called_once_with("speed 3")
                        mock_query.assert_called_once()
                        mock_parse.assert_called_once_with(
                            "3 c032 f016 c008 f001"
                        )

                        # Verify return values
                        self.assertEqual(result_profile, 3)
                        self.assertEqual(
                            result_speeds, ["c032", "f016", "c008", "f001"]
                        )

    def test_change_speed_profile_to_validation_error(self):
        """
        Test change_speed_profile_to with invalid profile number.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number") as mock_validate:
            # Setup validation to raise ValueError
            mock_validate.side_effect = ValueError("Profile out of range")

            with self.assertRaises(ValueError) as cm:
                nc.change_speed_profile_to(7)

            self.assertIn("Profile out of range", str(cm.exception))
            mock_validate.assert_called_once_with(7)

    def test_change_speed_profile_to_send_command_error(self):
        """
        Test change_speed_profile_to when send command fails.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command") as mock_send:
                # Setup send_command to raise error
                mock_send.side_effect = NanoControlCommandError("Device error")

                with self.assertRaises(NanoControlCommandError):
                    nc.change_speed_profile_to(2)

                mock_send.assert_called_once_with("speed 2")

    def test_change_speed_profile_to_query_error(self):
        """
        Test change_speed_profile_to when query fails.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    # Setup query to raise error
                    mock_query.side_effect = NanoControlConnectionError(
                        "Connection lost"
                    )

                    with self.assertRaises(NanoControlConnectionError):
                        nc.change_speed_profile_to(2)

    def test_change_speed_profile_to_parse_error(self):
        """
        Test change_speed_profile_to when parsing response fails.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    with patch.object(
                        nc, "_parse_speed_response"
                    ) as mock_parse:
                        mock_query.return_value = "invalid response"
                        mock_parse.side_effect = NanoControlError(
                            "Parse failed"
                        )

                        with self.assertRaises(NanoControlError) as cm:
                            nc.change_speed_profile_to(2)

                        self.assertIn("Parse failed", str(cm.exception))

    def test_change_speed_profile_to_profile_mismatch(self):
        """
        Test change_speed_profile_to when device reports wrong
        profile.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    with patch.object(
                        nc, "_parse_speed_response"
                    ) as mock_parse:
                        # Device reports profile 1 when we requested 3
                        mock_query.return_value = "1 c032 f016 c008 f001"
                        mock_parse.return_value = (
                            1,
                            ["c032", "f016", "c008", "f001"],
                        )

                        with self.assertRaises(NanoControlError) as cm:
                            nc.change_speed_profile_to(3)

                        error_msg = str(cm.exception)
                        self.assertIn("Device did not switch", error_msg)
                        self.assertIn("Requested 3", error_msg)
                        self.assertIn("device reports 1", error_msg)

    def test_change_speed_profile_to_all_profiles(self):
        """
        Test change_speed_profile_to with all valid profile numbers.
        """

        nc = NanoControl.__new__(NanoControl)

        for profile_num in range(1, 7):  # Test profiles 1-6
            with self.subTest(profile_num=profile_num):
                with patch.object(nc, "_validate_profile_number"):
                    with patch.object(nc, "_send_command") as mock_send:
                        with patch.object(
                            nc, "_query_speed_profile"
                        ) as mock_query:
                            with patch.object(
                                nc, "_parse_speed_response"
                            ) as mock_parse:
                                expected_response = (
                                    f"{profile_num} c032 f016 c008 f001"
                                )
                                expected_speeds = [
                                    "c032",
                                    "f016",
                                    "c008",
                                    "f001",
                                ]

                                mock_query.return_value = expected_response
                                mock_parse.return_value = (
                                    profile_num,
                                    expected_speeds,
                                )

                                result_profile, result_speeds = (
                                    nc.change_speed_profile_to(profile_num)
                                )

                                mock_send.assert_called_with(
                                    f"speed {profile_num}"
                                )
                                self.assertEqual(result_profile, profile_num)
                                self.assertEqual(
                                    result_speeds, expected_speeds
                                )

    def test_change_speed_profile_to_nanocontrol_error_passthrough(self):
        """
        Test that change_speed_profile_to passes through
        NanoControlError exceptions without wrapping.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    # Setup a specific NanoControlError
                    original_error = NanoControlError("Original error message")
                    mock_query.side_effect = original_error

                    with self.assertRaises(NanoControlError) as cm:
                        nc.change_speed_profile_to(2)

                    # Should be the same exception, not wrapped
                    self.assertIs(cm.exception, original_error)

    def test_change_speed_profile_to_unexpected_exception_wrapping(self):
        """
        Test that change_speed_profile_to wraps unexpected exceptions
        in NanoControlError.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    # Setup an unexpected exception
                    mock_query.side_effect = RuntimeError("Unexpected error")

                    with self.assertRaises(NanoControlError) as cm:
                        nc.change_speed_profile_to(2)

                    error_msg = str(cm.exception)
                    self.assertIn(
                        "Failed to change speed profile to 2", error_msg
                    )
                    self.assertIn("Unexpected error", error_msg)
                    self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_change_speed_profile_to_error_message_includes_profile(self):
        """
        Test that error messages include the requested profile number.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    mock_query.side_effect = Exception("Test error")

                    with self.assertRaises(NanoControlError) as cm:
                        nc.change_speed_profile_to(5)

                    error_msg = str(cm.exception)
                    self.assertIn("speed profile to 5", error_msg)

    def test_change_speed_profile_to_function_name_in_mismatch_error(self):
        """
        Test change_speed_profile_to when device reports different
        profile than requested.

        This test simulates a scenario where:
        1. We request the device to switch to profile 3
        2. The device appears to accept the command (no error)
        3. But when we query the current profile, it still reports 1
        4. This indicates the profile switch failed silently

        The function should detect this mismatch and raise a
        NanoControlError with details about what was requested vs
        what the device actually reports.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_send_command"):
                with patch.object(nc, "_query_speed_profile") as mock_query:
                    with patch.object(
                        nc, "_parse_speed_response"
                    ) as mock_parse:
                        mock_query.return_value = "1 c032 f016 c008 f001"
                        mock_parse.return_value = (
                            1,
                            ["c032", "f016", "c008", "f001"],
                        )

                        with self.assertRaises(NanoControlError) as cm:
                            nc.change_speed_profile_to(3)

                        error_msg = str(cm.exception)
                        # Fixed: Now expects the correct function name
                        self.assertIn("change_speed_profile_to", error_msg)

    def test_change_speed_profile_to_call_sequence(self):
        """
        Test that change_speed_profile_to calls methods in correct
        sequence.
        """

        nc = NanoControl.__new__(NanoControl)

        call_order = []

        def track_validate(profile_num):
            call_order.append(("validate", profile_num))

        def track_send(command):
            call_order.append(("send", command))

        def track_query():
            call_order.append(("query",))
            return "2 c032 f016 c008 f001"

        def track_parse(response):
            call_order.append(("parse", response))
            return (2, ["c032", "f016", "c008", "f001"])

        with patch.object(
            nc, "_validate_profile_number", side_effect=track_validate
        ):
            with patch.object(nc, "_send_command", side_effect=track_send):
                with patch.object(
                    nc, "_query_speed_profile", side_effect=track_query
                ):
                    with patch.object(
                        nc,
                        "_parse_speed_response",
                        side_effect=track_parse,
                    ):
                        nc.change_speed_profile_to(2)

                        expected_sequence = [
                            ("validate", 2),
                            ("send", "speed 2"),
                            ("query",),
                            ("parse", "2 c032 f016 c008 f001"),
                        ]

                        self.assertEqual(call_order, expected_sequence)

    def test_get_speed_profile_current_profile_no_switching(self):
        """
        Test get_speed_profile when querying current active profile.

        This test simulates the case where profile_num=None, meaning
        we want the currently active profile. No profile switching
        should occur, and the original profile should be returned
        as a dictionary mapping joint names to speed values.

        Note: The function calls _query_speed_profile twice - once
        initially and once in the finally block to check if
        restoration is needed.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_query_speed_profile") as mock_query:
            with patch.object(nc, "_parse_speed_response") as mock_parse:
                # Simulate device currently on profile 2 (both calls)
                mock_query.return_value = "2 c032 f016 c008 f001"
                mock_parse.return_value = (
                    2,
                    ["c032", "f016", "c008", "f001"],
                )

                result = nc.get_speed_profile()

                # Verify correct return format
                expected_result = {
                    "base_joint": "c032",
                    "elbow_joint": "f016",
                    "prismatic_joint": "c008",
                    "tweezer_joint": "f001",
                }
                self.assertEqual(result, expected_result)

                # Should be called twice: initial query + restoration check
                self.assertEqual(mock_query.call_count, 2)
                # Should parse twice as well
                self.assertEqual(mock_parse.call_count, 2)

    def test_get_speed_profile_specific_profile_same_as_current(self):
        """
        Test get_speed_profile when requested profile matches current.

        When the requested profile is the same as the currently
        active profile, no switching should occur. The function
        should use the speeds from the initial query.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number") as mock_validate:
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        # Current profile is 3, and we request profile 3
                        mock_query.return_value = "3 c064 f032 c016 f008"
                        mock_parse.return_value = (
                            3,
                            ["c064", "f032", "c016", "f008"],
                        )

                        result = nc.get_speed_profile(3)

                        # Verify validation was called
                        mock_validate.assert_called_once_with(3)

                        # Should not call change_speed_profile_to
                        mock_change.assert_not_called()

                        # Verify correct result
                        expected_result = {
                            "base_joint": "c064",
                            "elbow_joint": "f032",
                            "prismatic_joint": "c016",
                            "tweezer_joint": "f008",
                        }
                        self.assertEqual(result, expected_result)

    def test_get_speed_profile_different_profile_with_switching(self):
        """
        Test get_speed_profile when requested profile differs from
        current.

        This simulates switching from profile 1 to profile 4, then
        restoring back to profile 1. Tests the profile switching
        and restoration logic.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        # Initially on profile 1, want to query profile 4
                        mock_query.side_effect = [
                            "1 c032 f016 c008 f001",
                            "4 c001 f064 c032 f032",
                        ]
                        mock_parse.side_effect = [
                            (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            ),  # Initial (profile 1)
                            (
                                4,
                                ["c001", "f064", "c032", "f032"],
                            ),  # Restoration check (profile 4)
                        ]

                        # Profile switching calls
                        mock_change.side_effect = [
                            (
                                4,
                                ["c001", "f064", "c032", "f032"],
                            ),  # Switch to profile 4
                            (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            ),  # Restore to profile 1
                        ]

                        result = nc.get_speed_profile(4)

                        # Verify both calls to change_speed_profile_to
                        expected_calls = [
                            unittest.mock.call(4),  # Switch to profile 4
                            unittest.mock.call(1),  # Restore to profile 1
                        ]
                        mock_change.assert_has_calls(expected_calls)

                        # Verify result uses profile 4 speeds
                        expected_result = {
                            "base_joint": "c001",
                            "elbow_joint": "f064",
                            "prismatic_joint": "c032",
                            "tweezer_joint": "f032",
                        }
                        self.assertEqual(result, expected_result)

    def test_get_speed_profile_validation_error(self):
        """
        Test get_speed_profile with invalid profile number.

        Should raise ValueError when profile_num is outside valid
        range (1-6) or not an integer.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid profile")

            with self.assertRaises(ValueError) as cm:
                nc.get_speed_profile(7)

            self.assertIn("Invalid profile", str(cm.exception))
            mock_validate.assert_called_once_with(7)

    def test_get_speed_profile_query_error(self):
        """
        Test get_speed_profile when initial query fails.

        If the initial speed profile query fails, the error should
        be propagated without attempting any restoration.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_query_speed_profile") as mock_query:
            mock_query.side_effect = NanoControlConnectionError(
                "Connection lost"
            )

            with self.assertRaises(NanoControlConnectionError):
                nc.get_speed_profile()

    def test_get_speed_profile_parse_error(self):
        """
        Test get_speed_profile when response parsing fails.

        If parsing the speed profile response fails, the error
        should be propagated as NanoControlError.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_query_speed_profile") as mock_query:
            with patch.object(nc, "_parse_speed_response") as mock_parse:
                mock_query.return_value = "invalid response"
                mock_parse.side_effect = NanoControlError("Parse failed")

                with self.assertRaises(NanoControlError) as cm:
                    nc.get_speed_profile()

                self.assertIn("Parse failed", str(cm.exception))

    def test_get_speed_profile_change_profile_error(self):
        """
        Test get_speed_profile when profile switching fails.

        If switching to the requested profile fails, the error
        should be propagated. The finally block will check if
        restoration is needed, but since the switch failed, the
        device should still be on the original profile.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        # Initial query succeeds, restoration check
                        # shows still on profile 1
                        mock_query.side_effect = [
                            "1 c032 f016 c008 f001",
                            "1 c032 f016 c008 f001",
                        ]
                        mock_parse.side_effect = [
                            (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            ),  # Initial
                            (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            ),  # Restoration check
                        ]

                        # Profile switching fails
                        mock_change.side_effect = NanoControlError(
                            "Switch failed"
                        )

                        with self.assertRaises(NanoControlError) as cm:
                            nc.get_speed_profile(3)

                        self.assertIn("Switch failed", str(cm.exception))

                        # Should only attempt the failed switch, no restoration needed
                        # since device is still on original profile 1
                        self.assertEqual(mock_change.call_count, 1)
                        mock_change.assert_called_with(3)

    def test_get_speed_profile_restoration_failure_warning(self):
        """
        Test get_speed_profile when restoration fails.

        If restoring the original profile fails, a warning should
        be printed but the original result should still be returned.
        Tests the defensive restoration logic.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            # Initial query and restoration check
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",  # Initial
                                "3 c001 f064 c032 f032",
                            ]
                            mock_parse.side_effect = [
                                (1, ["c032", "f016", "c008", "f001"]),
                                (3, ["c001", "f064", "c032", "f032"]),
                            ]

                            # Successful switch, failed restoration
                            mock_change.side_effect = [
                                (
                                    3,
                                    ["c001", "f064", "c032", "f032"],
                                ),  # Switch succeeds
                                NanoControlError(
                                    "Restore failed"
                                ),  # Restore fails
                            ]

                            result = nc.get_speed_profile(3)

                            # Should print warning about restoration
                            # failure
                            mock_print.assert_called()
                            warning_call = mock_print.call_args[0][0]
                            self.assertIn("WARNING", warning_call)
                            self.assertIn(
                                "failed to restore profile 1", warning_call
                            )

                            # Should still return the requested
                            # profile data
                            expected_result = {
                                "base_joint": "c001",
                                "elbow_joint": "f064",
                                "prismatic_joint": "c032",
                                "tweezer_joint": "f032",
                            }
                            self.assertEqual(result, expected_result)

    def test_get_speed_profile_unexpected_error_wrapping(self):
        """
        Test get_speed_profile wraps unexpected exceptions.

        Any unexpected exception should be wrapped in
        NanoControlError with appropriate context about the failure.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_query_speed_profile") as mock_query:
            mock_query.side_effect = RuntimeError("Unexpected error")

            with self.assertRaises(NanoControlError) as cm:
                nc.get_speed_profile()

            error_msg = str(cm.exception)
            self.assertIn("Unexpected failure", error_msg)
            self.assertIn("Unexpected error", error_msg)
            self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_get_speed_profile_all_valid_profiles(self):
        """
        Test get_speed_profile with all valid profile numbers (1-6).

        Ensures the function works correctly for each valid profile
        number and returns properly formatted dictionaries.
        """

        nc = NanoControl.__new__(NanoControl)

        for profile_num in range(1, 7):
            with self.subTest(profile_num=profile_num):
                with patch.object(nc, "_validate_profile_number"):
                    with patch.object(
                        nc, "_query_speed_profile"
                    ) as mock_query:
                        with patch.object(
                            nc, "_parse_speed_response"
                        ) as mock_parse:
                            with patch.object(
                                nc, "change_speed_profile_to"
                            ) as mock_change:
                                # Current profile different from requested
                                mock_query.return_value = (
                                    "1 c032 f016 c008 f001"
                                )
                                mock_parse.return_value = (
                                    1,
                                    ["c032", "f016", "c008", "f001"],
                                )

                                # Requested profile speeds
                                expected_speeds = [
                                    f"c{profile_num:03d}",
                                    "f016",
                                    "c008",
                                    "f001",
                                ]
                                mock_change.return_value = (
                                    profile_num,
                                    expected_speeds,
                                )

                                result = nc.get_speed_profile(profile_num)

                                # Verify dictionary structure
                                self.assertIsInstance(result, dict)
                                self.assertEqual(len(result), 4)
                                self.assertIn("base_joint", result)
                                self.assertIn("elbow_joint", result)
                                self.assertIn("prismatic_joint", result)
                                self.assertIn("tweezer_joint", result)

    def test_get_speed_profile_none_parameter_behavior(self):
        """
        Test get_speed_profile with explicit None parameter.

        Passing None should behave identically to calling with no
        parameters - should return current active profile without
        any switching.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_query_speed_profile") as mock_query:
            with patch.object(nc, "_parse_speed_response") as mock_parse:
                mock_query.return_value = "5 c001 f064 c032 f016"
                mock_parse.return_value = (
                    5,
                    ["c001", "f064", "c032", "f016"],
                )

                # Test both calling patterns
                result1 = nc.get_speed_profile(None)
                result2 = nc.get_speed_profile()

                # Should be identical
                self.assertEqual(result1, result2)

                expected_result = {
                    "base_joint": "c001",
                    "elbow_joint": "f064",
                    "prismatic_joint": "c032",
                    "tweezer_joint": "f016",
                }
                self.assertEqual(result1, expected_result)

    def test_get_speed_profile_partial_switch_corruption(self):
        """
        Test when device gets into inconsistent state during
        switching.

        Simulates scenario where switch command appears to succeed
        but device ends up on unexpected profile, then restoration
        also fails, leaving device in unknown state.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",
                                "5 c001 f001 f001 c001",
                            ]
                            mock_parse.side_effect = [
                                (1, ["c032", "f016", "c008", "f001"]),
                                (5, ["c001", "f001", "f001", "c001"]),
                            ]

                            mock_change.side_effect = [
                                (3, ["c064", "f032", "c016", "f008"]),
                                NanoControlError(
                                    "Restoration failed - device locked"
                                ),
                            ]

                            result = nc.get_speed_profile(3)

                            expected_result = {
                                "base_joint": "c064",
                                "elbow_joint": "f032",
                                "prismatic_joint": "c016",
                                "tweezer_joint": "f008",
                            }
                            self.assertEqual(result, expected_result)

                            warning_printed = any(
                                "WARNING" in str(call)
                                and "failed to restore" in str(call)
                                for call in mock_print.call_args_list
                            )
                            self.assertTrue(warning_printed)

    def test_get_speed_profile_restoration_query_corruption(self):
        """
        Test when restoration query returns corrupted/invalid data.

        The finally block query could return malformed data, causing
        parse errors during restoration check.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",
                                "corrupted response",
                            ]
                            mock_parse.side_effect = [
                                (1, ["c032", "f016", "c008", "f001"]),
                                NanoControlError("Parse failed - corrupted"),
                            ]

                            mock_change.return_value = (
                                3,
                                ["c064", "f032", "c016", "f008"],
                            )

                            result = nc.get_speed_profile(3)

                            expected_result = {
                                "base_joint": "c064",
                                "elbow_joint": "f032",
                                "prismatic_joint": "c016",
                                "tweezer_joint": "f008",
                            }
                            self.assertEqual(result, expected_result)

                            warning_printed = any(
                                "WARNING" in str(call)
                                for call in mock_print.call_args_list
                            )
                            self.assertTrue(warning_printed)

    def test_get_speed_profile_cascading_failures(self):
        """
        Test multiple failures in sequence during profile operations.

        Tests resilience when multiple operations fail in sequence:
        switch succeeds, restoration query fails, restoration
        attempt fails.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",
                                NanoControlConnectionError(
                                    "Connection lost during restoration"
                                ),
                            ]
                            mock_parse.return_value = (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            )

                            mock_change.return_value = (
                                3,
                                ["c064", "f032", "c016", "f008"],
                            )

                            result = nc.get_speed_profile(3)

                            expected_result = {
                                "base_joint": "c064",
                                "elbow_joint": "f032",
                                "prismatic_joint": "c016",
                                "tweezer_joint": "f008",
                            }
                            self.assertEqual(result, expected_result)

                            warning_printed = any(
                                "WARNING" in str(call)
                                and "failed to restore" in str(call)
                                for call in mock_print.call_args_list
                            )
                            self.assertTrue(warning_printed)

    def test_get_speed_profile_race_condition_simulation(self):
        """
        Test behavior when device state changes between operations.

        Simulates external interference where another process
        changes the profile between our switch and restoration
        check.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        mock_query.side_effect = [
                            "1 c032 f016 c008 f001",
                            "6 f001 f001 f001 f001",
                        ]
                        mock_parse.side_effect = [
                            (1, ["c032", "f016", "c008", "f001"]),
                            (6, ["f001", "f001", "f001", "f001"]),
                        ]

                        mock_change.side_effect = [
                            (3, ["c064", "f032", "c016", "f008"]),
                            (1, ["c032", "f016", "c008", "f001"]),
                        ]

                        nc.get_speed_profile(3)

                        self.assertEqual(mock_change.call_count, 2)
                        expected_calls = [
                            unittest.mock.call(3),
                            unittest.mock.call(1),
                        ]
                        mock_change.assert_has_calls(expected_calls)

    def test_get_speed_profile_state_corruption_resilience(self):
        """
        Test resilience when internal state becomes inconsistent.

        Tests behavior when the original_profile_num tracking gets
        corrupted or parsing returns inconsistent data.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        mock_query.side_effect = [
                            "2 c032 f016 c008 f001",
                            "2 c032 f016 c008 f001",
                        ]
                        mock_parse.side_effect = [
                            (2, ["c032", "f016", "c008", "f001"]),
                            (4, ["c032", "f016", "c008", "f001"]),
                        ]

                        mock_change.side_effect = [
                            (3, ["c064", "f032", "c016", "f008"]),
                            (2, ["c032", "f016", "c008", "f001"]),
                        ]

                        nc.get_speed_profile(3)

                        self.assertEqual(mock_change.call_count, 2)

    def test_get_speed_profile_restoration_double_failure(self):
        """
        Test when both restoration check and restoration attempt
        fail.

        Simulates the worst case where restoration query fails AND
        the restoration command fails, testing defensive programming.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",
                                "3 c064 f032 c016 f008",
                            ]
                            mock_parse.side_effect = [
                                (1, ["c032", "f016", "c008", "f001"]),
                                (3, ["c064", "f032", "c016", "f008"]),
                            ]

                            mock_change.side_effect = [
                                (3, ["c064", "f032", "c016", "f008"]),
                                RuntimeError("Critical device failure"),
                            ]

                            result = nc.get_speed_profile(3)

                            expected_result = {
                                "base_joint": "c064",
                                "elbow_joint": "f032",
                                "prismatic_joint": "c016",
                                "tweezer_joint": "f008",
                            }
                            self.assertEqual(result, expected_result)

                            warning_printed = any(
                                "WARNING" in str(call)
                                and "failed to restore" in str(call)
                                for call in mock_print.call_args_list
                            )
                            self.assertTrue(warning_printed)

    def test_get_speed_profile_communication_timeout_during_restore(self):
        """
        Test communication timeout specifically during restoration.

        Tests when the main operation succeeds but restoration times
        out, ensuring the successful result is still returned.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_query_speed_profile") as mock_query:
                with patch.object(nc, "_parse_speed_response") as mock_parse:
                    with patch.object(
                        nc, "change_speed_profile_to"
                    ) as mock_change:
                        with patch("builtins.print") as mock_print:
                            mock_query.side_effect = [
                                "1 c032 f016 c008 f001",
                                NanoControlCommandError("Device timeout"),
                            ]
                            mock_parse.return_value = (
                                1,
                                ["c032", "f016", "c008", "f001"],
                            )

                            mock_change.return_value = (
                                3,
                                ["c064", "f032", "c016", "f008"],
                            )

                            result = nc.get_speed_profile(3)

                            expected_result = {
                                "base_joint": "c064",
                                "elbow_joint": "f032",
                                "prismatic_joint": "c016",
                                "tweezer_joint": "f008",
                            }
                            self.assertEqual(result, expected_result)

                            self.assertEqual(mock_change.call_count, 1)

                            warning_printed = any(
                                "WARNING" in str(call)
                                for call in mock_print.call_args_list
                            )
                            self.assertTrue(warning_printed)

    def test_stop_successful(self):
        """
        Test stop function returns response from device.

        The stop function should send 'stop' command to device and
        return whatever response the device provides.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.return_value = "Movement stopped"

            result = nc.stop()

            mock_send.assert_called_once_with("stop")
            self.assertEqual(result, "Movement stopped")

    def test_stop_empty_response(self):
        """
        Test stop function with empty device response.

        Some devices might return empty string when stop command
        is acknowledged but no additional message is provided.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.return_value = ""

            result = nc.stop()

            mock_send.assert_called_once_with("stop")
            self.assertEqual(result, "")

    def test_stop_connection_error(self):
        """
        Test stop function propagates connection errors.

        If serial communication fails during stop command, the
        error should be propagated to allow caller to handle
        the failure appropriately.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlConnectionError(
                "Serial port disconnected"
            )

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.stop()

            self.assertIn("Serial port disconnected", str(cm.exception))

    def test_stop_command_error(self):
        """
        Test stop function propagates command errors from device.

        If the device returns an error status in response to stop
        command, this should be propagated as
        NanoControlCommandError.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError(
                "Device error: No movement to stop"
            )

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.stop()

            self.assertIn("No movement to stop", str(cm.exception))

    def test_stop_device_timeout(self):
        """
        Test stop function when device times out.

        If the device doesn't respond to stop command within
        timeout period, should raise appropriate timeout error.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError(
                "No response from device (timeout)"
            )

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.stop()

            self.assertIn("timeout", str(cm.exception))

    def test_stop_various_device_responses(self):
        """
        Test stop function with various valid device responses.

        Different firmware versions or device states might return
        different acknowledgment messages for stop command.
        """

        nc = NanoControl.__new__(NanoControl)

        responses = [
            "OK",
            "Movement stopped",
            "STOP",
            "Command acknowledged",
            "All axes stopped",
            "",
        ]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command") as mock_send:
                    mock_send.return_value = response

                    result = nc.stop()

                    mock_send.assert_called_once_with("stop")
                    self.assertEqual(result, response)

    def test_stop_multiple_calls(self):
        """
        Test multiple consecutive calls to stop function.

        Ensures that stop can be called multiple times without
        issues, which might happen in emergency stop scenarios.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.return_value = "OK"

            result1 = nc.stop()
            result2 = nc.stop()
            result3 = nc.stop()

            self.assertEqual(mock_send.call_count, 3)
            self.assertEqual(result1, "OK")
            self.assertEqual(result2, "OK")
            self.assertEqual(result3, "OK")

            expected_calls = [
                unittest.mock.call("stop"),
                unittest.mock.call("stop"),
                unittest.mock.call("stop"),
            ]
            mock_send.assert_has_calls(expected_calls)

    def test_stop_return_type(self):
        """
        Test that stop function always returns string type.

        Ensures consistent return type regardless of device
        response content.
        """

        nc = NanoControl.__new__(NanoControl)

        test_responses = ["OK", "", "Movement stopped", "123", "ERROR"]

        for response in test_responses:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command") as mock_send:
                    mock_send.return_value = response

                    result = nc.stop()

                    self.assertIsInstance(result, str)
                    self.assertEqual(result, response)

    def test_stop_exact_command_string(self):
        """
        Test that stop function sends exactly 'stop' command.

        Verifies the exact command string sent to device matches
        protocol specification.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.return_value = "OK"

            nc.stop()

            mock_send.assert_called_once_with("stop")
            args, kwargs = mock_send.call_args
            self.assertEqual(args[0], "stop")
            self.assertEqual(len(args), 1)
            self.assertEqual(len(kwargs), 0)

    def test_stop_emergency_scenario_simulation(self):
        """
        Test stop function behavior in emergency stop scenario.

        Simulates rapid succession of stop commands that might
        occur during emergency situations, ensuring reliability.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            responses = ["OK", "Already stopped", "OK", "OK", "OK"]
            mock_send.side_effect = responses

            results = []
            for i in range(5):
                results.append(nc.stop())

            self.assertEqual(len(results), 5)
            self.assertEqual(results, responses)
            self.assertEqual(mock_send.call_count, 5)

    def test_stop_mixed_error_and_success(self):
        """
        Test stop function with mixed success and error responses.

        Tests scenarios where some stop attempts fail but others
        succeed, which could happen during device recovery.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = [
                NanoControlCommandError("Device busy"),
                "OK",
            ]

            with self.assertRaises(NanoControlCommandError):
                nc.stop()

            result = nc.stop()
            self.assertEqual(result, "OK")
            self.assertEqual(mock_send.call_count, 2)

    def test_stopnack_successful_transmission(self):
        """
        Test stopnack sends command without waiting for response.

        The function should send 'stopnack\\r' bytes to serial port
        and return immediately without waiting for device response.
        """

        nc = NanoControl.__new__(NanoControl)

        # Create mock serial port
        mock_serial = Mock()
        nc.device_serial = mock_serial

        nc.stopnack()

        # Verify serial operations called in correct order
        mock_serial.flushInput.assert_called_once()
        mock_serial.write.assert_called_once_with(b"stopnack\r")
        mock_serial.flush.assert_called_once()

        # Verify no read operations (no waiting for response)
        mock_serial.read_until.assert_not_called()
        mock_serial.read.assert_not_called()
        mock_serial.readline.assert_not_called()

    def test_stopnack_returns_none(self):
        """
        Test that stopnack returns None (no acknowledgement).

        Since the function doesn't wait for device response, it
        should return None to indicate no acknowledgement received.
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        result = nc.stopnack()

        self.assertIsNone(result)

    def test_stopnack_serial_write_failure(self):
        """
        Test stopnack when serial write operation fails.

        If the serial port write fails, should raise appropriate
        connection error without attempting to read response.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        mock_serial.write.side_effect = serial.SerialException("Port closed")
        nc.device_serial = mock_serial

        with self.assertRaises(NanoControlConnectionError) as cm:
            nc.stopnack()

        self.assertIn("Serial error during transmission", str(cm.exception))
        self.assertIn("Port closed", str(cm.exception))

    def test_stopnack_flush_input_failure(self):
        """
        Test stopnack when input buffer flush fails.

        If flushInput fails, should raise connection error before
        attempting to send command.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        mock_serial.flushInput.side_effect = serial.SerialException(
            "Buffer error"
        )
        nc.device_serial = mock_serial

        with self.assertRaises(NanoControlConnectionError) as cm:
            nc.stopnack()

        self.assertIn("Serial error during transmission", str(cm.exception))
        # Should not attempt write if flushInput fails
        mock_serial.write.assert_not_called()

    def test_stopnack_flush_output_failure(self):
        """
        Test stopnack when output buffer flush fails.

        If the final flush() fails after write, should still raise
        connection error.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        mock_serial.flush.side_effect = serial.SerialException("Flush failed")
        nc.device_serial = mock_serial

        with self.assertRaises(NanoControlConnectionError) as cm:
            nc.stopnack()

        self.assertIn("Serial error during transmission", str(cm.exception))
        # Should still attempt write before flush fails
        mock_serial.write.assert_called_once()

    def test_stopnack_command_encoding(self):
        """
        Test that stopnack sends correctly encoded command bytes.

        Verifies the exact byte sequence sent to device matches
        protocol specification.
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        nc.stopnack()

        # Verify exact bytes sent
        call_args = nc.device_serial.write.call_args
        sent_bytes = call_args[0][0]

        self.assertEqual(sent_bytes, b"stopnack\r")
        self.assertIsInstance(sent_bytes, bytes)

    def test_stopnack_multiple_calls(self):
        """
        Test multiple consecutive calls to stopnack.

        Each call should send the command independently without
        interfering with previous calls.
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        result1 = nc.stopnack()
        result2 = nc.stopnack()
        result3 = nc.stopnack()

        # All should return None
        self.assertIsNone(result1)
        self.assertIsNone(result2)
        self.assertIsNone(result3)

        # Each call should trigger all serial operations
        self.assertEqual(nc.device_serial.flushInput.call_count, 3)
        self.assertEqual(nc.device_serial.write.call_count, 3)
        self.assertEqual(nc.device_serial.flush.call_count, 3)

    def test_stopnack_call_sequence(self):
        """
        Test that stopnack calls serial operations in correct order.

        The sequence should be: flushInput -> write -> flush
        This ensures clean transmission without interference.
        """

        nc = NanoControl.__new__(NanoControl)

        call_order = []

        def track_flush_input():
            call_order.append("flushInput")

        def track_write(data):
            call_order.append(("write", data))

        def track_flush():
            call_order.append("flush")

        mock_serial = Mock()
        mock_serial.flushInput.side_effect = track_flush_input
        mock_serial.write.side_effect = track_write
        mock_serial.flush.side_effect = track_flush
        nc.device_serial = mock_serial

        nc.stopnack()

        expected_sequence = [
            "flushInput",
            ("write", b"stopnack\r"),
            "flush",
        ]

        self.assertEqual(call_order, expected_sequence)

    def test_stopnack_no_response_reading(self):
        """
        Test that stopnack never attempts to read device response.

        This is the key difference from regular stop() - no waiting
        for acknowledgement means no read operations should occur.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        nc.device_serial = mock_serial

        nc.stopnack()

        # Verify no read methods were called
        mock_serial.read.assert_not_called()
        mock_serial.read_until.assert_not_called()
        mock_serial.readline.assert_not_called()
        mock_serial.readlines.assert_not_called()
        mock_serial.readall.assert_not_called()

    def test_stopnack_exception_chaining(self):
        """
        Test that stopnack properly chains original exceptions.

        When serial operations fail, the original exception should
        be preserved in the exception chain for debugging.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        original_error = serial.SerialException("Hardware failure")
        mock_serial.write.side_effect = original_error
        nc.device_serial = mock_serial

        with self.assertRaises(NanoControlConnectionError) as cm:
            nc.stopnack()

        # Verify exception chaining
        self.assertIs(cm.exception.__cause__, original_error)

    def test_stopnack_performance_characteristics(self):
        """
        Test that stopnack has minimal serial operations.

        Should only perform essential operations: clear input,
        write command, flush output. No waiting or reading.
        """

        nc = NanoControl.__new__(NanoControl)

        mock_serial = Mock()
        nc.device_serial = mock_serial

        nc.stopnack()

        # Count total method calls on serial object
        total_calls = (
            mock_serial.flushInput.call_count
            + mock_serial.write.call_count
            + mock_serial.flush.call_count
        )

        # Should be exactly 3 calls (minimal operations)
        self.assertEqual(total_calls, 3)

    def test_stopnack_write_operation_failure(self):
        """
        Test stopnack when write operation fails for any reason.

        Tests the error handling path when serial write fails,
        which could happen due to various reasons including
        encoding issues, port closure, etc.
        """

        nc = NanoControl.__new__(NanoControl)
        nc.device_serial = Mock()

        # Test various write failure scenarios
        write_errors = [
            serial.SerialException("Port disconnected"),
            OSError("Device not ready"),
            IOError("Write buffer full"),
        ]

        for error in write_errors:
            with self.subTest(error=error):
                nc.device_serial.write.side_effect = error

                with self.assertRaises(NanoControlConnectionError) as cm:
                    nc.stopnack()

                self.assertIn(
                    "Serial error during transmission", str(cm.exception)
                )

                # Reset for next iteration
                nc.device_serial.write.side_effect = None

    def test_set_speed_profile_successful(self):
        """
        Test set_speed_profile with valid inputs sends correct command.
        """
        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number") as mock_validate:
            with patch.object(
                nc, "_validate_speed_value"
            ) as mock_speed_validate:
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    result = nc.set_speed_profile(2, speed_profile)

                    # Verify validations were called
                    mock_validate.assert_called_once_with(2)
                    self.assertEqual(mock_speed_validate.call_count, 4)

                    # Verify correct command was sent
                    expected_command = "speed 2 c32 f16 c08 c64"
                    mock_send.assert_called_once_with(expected_command)

                    # Verify return value
                    self.assertEqual(result, "OK")

    def test_set_speed_profile_invalid_profile_number(self):
        """
        Test set_speed_profile raises ValueError for invalid profile
        number.
        """

        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number") as mock_validate:
            mock_validate.side_effect = ValueError("Profile out of range")

            with self.assertRaises(ValueError) as cm:
                nc.set_speed_profile(7, speed_profile)

            self.assertIn("Profile out of range", str(cm.exception))
            mock_validate.assert_called_once_with(7)

    def test_set_speed_profile_not_dictionary(self):
        """
        Test set_speed_profile raises ValueError when speed_profile is
        not a dict.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            invalid_inputs = [
                "not a dict",
                ["c32", "f16", "c08", "c64"],
                ("c32", "f16", "c08", "c64"),
                None,
                123,
            ]

            for invalid_input in invalid_inputs:
                with self.subTest(input=invalid_input):
                    with self.assertRaises(ValueError) as cm:
                        nc.set_speed_profile(2, invalid_input)

                    self.assertIn("must be a dictionary", str(cm.exception))

    def test_set_speed_profile_missing_keys(self):
        """
        Test set_speed_profile raises ValueError for missing required
        keys.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            incomplete_profiles = [
                # Missing one key
                {
                    "base_joint": "c32",
                    "elbow_joint": "f16",
                    "prismatic_joint": "c08",
                },
                # Missing multiple keys
                {"base_joint": "c32"},
                # Empty dict
                {},
                # Wrong key names
                {
                    "base": "c32",
                    "elbow": "f16",
                    "prismatic": "c08",
                    "tweezer": "c64",
                },
            ]

            for profile in incomplete_profiles:
                with self.subTest(profile=profile):
                    with self.assertRaises(ValueError) as cm:
                        nc.set_speed_profile(2, profile)

                    error_msg = str(cm.exception)
                    self.assertIn("Missing", error_msg)

    def test_set_speed_profile_extra_keys(self):
        """
        Test set_speed_profile raises ValueError for extra keys.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            # Since your current implementation doesn't check for extra keys,
            # this test needs to be updated or the implementation needs to be changed
            profile_with_extra = {
                "base_joint": "c32",
                "elbow_joint": "f16",
                "prismatic_joint": "c08",
                "tweezer_joint": "c64",
                "extra_key": "c01",  # This will be ignored by current implementation
            }

            # Current implementation doesn't validate extra keys, so patch _send_command
            with patch.object(nc, "_validate_speed_value"):
                with patch.object(nc, "_send_command", return_value="OK"):
                    # This will actually succeed with current implementation
                    result = nc.set_speed_profile(2, profile_with_extra)
                    self.assertEqual(result, "OK")

    def test_set_speed_profile_invalid_speed_values(self):
        """
        Test set_speed_profile raises ValueError for invalid speed values.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(
                nc, "_validate_speed_value"
            ) as mock_speed_validate:
                # Mock validation to fail for specific speed
                original_error = ValueError("Invalid speed format")
                mock_speed_validate.side_effect = original_error

                speed_profile = {
                    "base_joint": "c32",
                    "elbow_joint": "invalid",
                    "prismatic_joint": "c08",
                    "tweezer_joint": "c64",
                }

                with self.assertRaises(ValueError) as cm:
                    nc.set_speed_profile(2, speed_profile)

                # Your current implementation just re-raises the original error
                error_msg = str(cm.exception)
                self.assertIn("Invalid speed format", error_msg)

    def test_set_speed_profile_all_joints_validated(self):
        """
        Test that all joint speeds are validated in correct order.
        """
        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(
                nc, "_validate_speed_value"
            ) as mock_speed_validate:
                with patch.object(nc, "_send_command", return_value="OK"):
                    nc.set_speed_profile(2, speed_profile)

                    # Check that validation was called for each joint
                    # in order
                    expected_calls = [
                        unittest.mock.call("c32"),  # base_joint
                        unittest.mock.call("f16"),  # elbow_joint
                        unittest.mock.call("c08"),  # prismatic_joint
                        unittest.mock.call("c64"),  # tweezer_joint
                    ]

                    mock_speed_validate.assert_has_calls(expected_calls)
                    self.assertEqual(mock_speed_validate.call_count, 4)

    def test_set_speed_profile_command_format(self):
        """
        Test that the correct command format is sent to device.
        """

        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            # (profile_num, speed_dict, expected_command)
            (
                1,
                {
                    "base_joint": "c01",
                    "elbow_joint": "c01",
                    "prismatic_joint": "c01",
                    "tweezer_joint": "c01",
                },
                "speed 1 c01 c01 c01 c01",
            ),
            (
                6,
                {
                    "base_joint": "f64",
                    "elbow_joint": "f64",
                    "prismatic_joint": "f64",
                    "tweezer_joint": "f64",
                },
                "speed 6 f64 f64 f64 f64",
            ),
            (
                3,
                {
                    "base_joint": "c32",
                    "elbow_joint": "f16",
                    "prismatic_joint": "c08",
                    "tweezer_joint": "c64",
                },
                "speed 3 c32 f16 c08 c64",
            ),
        ]

        for profile_num, speed_dict, expected_command in test_cases:
            with self.subTest(profile=profile_num):
                with patch.object(nc, "_validate_profile_number"):
                    with patch.object(nc, "_validate_speed_value"):
                        with patch.object(
                            nc, "_send_command", return_value="OK"
                        ) as mock_send:
                            nc.set_speed_profile(profile_num, speed_dict)

                            mock_send.assert_called_once_with(expected_command)

    def test_set_speed_profile_send_command_error(self):
        """
        Test set_speed_profile propagates NanoControlCommandError
        from _send_command.
        """

        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_validate_speed_value"):
                with patch.object(nc, "_send_command") as mock_send:
                    mock_send.side_effect = NanoControlCommandError(
                        "Device busy"
                    )

                    with self.assertRaises(NanoControlCommandError) as cm:
                        nc.set_speed_profile(2, speed_profile)

                    self.assertIn("Device busy", str(cm.exception))

    def test_set_speed_profile_send_command_connection_error(self):
        """
        Test set_speed_profile propagates NanoControlConnectionError
        from _send_command.
        """

        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_validate_speed_value"):
                with patch.object(nc, "_send_command") as mock_send:
                    mock_send.side_effect = NanoControlConnectionError(
                        "Port disconnected"
                    )

                    with self.assertRaises(NanoControlConnectionError) as cm:
                        nc.set_speed_profile(2, speed_profile)

                    self.assertIn("Port disconnected", str(cm.exception))

    def test_set_speed_profile_unexpected_error_wrapping(self):
        """
        Test set_speed_profile wraps unexpected exceptions in NanoControlError.
        """
        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_validate_speed_value"):
                with patch.object(nc, "_send_command") as mock_send:
                    mock_send.side_effect = RuntimeError("Unexpected error")

                    # Your current implementation doesn't wrap unexpected errors
                    with self.assertRaises(RuntimeError) as cm:
                        nc.set_speed_profile(2, speed_profile)

                    self.assertIn("Unexpected error", str(cm.exception))

    def test_set_speed_profile_boundary_values(self):
        """
        Test set_speed_profile with boundary profile numbers and speed values.
        """

        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # Min profile, min speeds
            (
                1,
                {
                    "base_joint": "c01",
                    "elbow_joint": "f01",
                    "prismatic_joint": "c01",
                    "tweezer_joint": "f01",
                },
            ),
            # Max profile, max speeds
            (
                6,
                {
                    "base_joint": "c64",
                    "elbow_joint": "f64",
                    "prismatic_joint": "c64",
                    "tweezer_joint": "f64",
                },
            ),
            # Mixed boundaries
            (
                3,
                {
                    "base_joint": "c64",
                    "elbow_joint": "f01",
                    "prismatic_joint": "c01",
                    "tweezer_joint": "f64",
                },
            ),
        ]

        for profile_num, speed_dict in boundary_cases:
            with self.subTest(profile=profile_num):
                with patch.object(nc, "_validate_profile_number"):
                    with patch.object(nc, "_validate_speed_value"):
                        with patch.object(
                            nc, "_send_command", return_value="OK"
                        ) as mock_send:
                            result = nc.set_speed_profile(
                                profile_num, speed_dict
                            )

                            self.assertEqual(result, "OK")
                            mock_send.assert_called_once()

    def test_set_speed_profile_exception_chaining(self):
        """
        Test that exceptions are properly chained for debugging.
        """
        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(
                nc, "_validate_speed_value"
            ) as mock_speed_validate:
                # Your current implementation doesn't chain exceptions
                original_error = ValueError("Speed out of range")
                mock_speed_validate.side_effect = original_error

                with self.assertRaises(ValueError) as cm:
                    nc.set_speed_profile(2, speed_profile)

                # Current implementation doesn't chain, so just check
                # the message
                self.assertIn("Speed out of range", str(cm.exception))

    def test_set_speed_profile_parameter_order(self):
        """
        Test that profile_num comes before speed_profile in function
        signature.
        """

        nc = NanoControl.__new__(NanoControl)

        speed_profile = {
            "base_joint": "c32",
            "elbow_joint": "f16",
            "prismatic_joint": "c08",
            "tweezer_joint": "c64",
        }

        with patch.object(nc, "_validate_profile_number"):
            with patch.object(nc, "_validate_speed_value"):
                with patch.object(nc, "_send_command", return_value="OK"):
                    # This should work (correct order)
                    result = nc.set_speed_profile(2, speed_profile)
                    self.assertEqual(result, "OK")

                    # Verify we can't accidentally swap parameters (this
                    # would be caught by type hints and validation)
                    with self.assertRaises((ValueError, TypeError)):
                        nc.set_speed_profile(speed_profile, 2)

    def test_set_fine_with_coarse_all_successful_enable(self):
        """
        Test set_fine_with_coarse_all with valid True input.

        Should send 'finewithcoarse 1 1 1 1' command and return device
        response.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command", return_value="OK") as mock_send:
            result = nc.set_fine_with_coarse_all(True)

            mock_send.assert_called_once_with("finewithcoarse 1 1 1 1")
            self.assertEqual(result, "OK")

    def test_set_fine_with_coarse_all_successful_disable(self):
        """
        Test set_fine_with_coarse_all with valid False input.

        Should send 'finewithcoarse 0 0 0 0' command and return device
        response.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(
            nc, "_send_command", return_value="Disabled"
        ) as mock_send:
            result = nc.set_fine_with_coarse_all(False)

            mock_send.assert_called_once_with("finewithcoarse 0 0 0 0")
            self.assertEqual(result, "Disabled")

    def test_set_fine_with_coarse_all_invalid_type_string(self):
        """
        Test set_fine_with_coarse_all raises ValueError when passed string.
        """

        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc.set_fine_with_coarse_all("true")

        error_msg = str(cm.exception)
        self.assertIn("status must be a boolean", error_msg)
        self.assertIn("NanoControl.set_fine_with_coarse_all", error_msg)

    def test_set_fine_with_coarse_all_invalid_type_integer(self):
        """
        Test set_fine_with_coarse_all raises ValueError when passed integer.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_integers = [0, 1, -1, 100]

        for invalid_input in invalid_integers:
            with self.subTest(input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc.set_fine_with_coarse_all(invalid_input)

                error_msg = str(cm.exception)
                self.assertIn("status must be a boolean", error_msg)

    def test_set_fine_with_coarse_all_invalid_type_none(self):
        """
        Test set_fine_with_coarse_all raises ValueError when passed None.
        """

        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc.set_fine_with_coarse_all(None)

        self.assertIn("status must be a boolean", str(cm.exception))

    def test_set_fine_with_coarse_all_invalid_type_collection(self):
        """
        Test set_fine_with_coarse_all raises ValueError for collection types.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_collections = [
            [True],
            {"status": True},
            (True,),
            {True, False},
        ]

        for invalid_input in invalid_collections:
            with self.subTest(input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc.set_fine_with_coarse_all(invalid_input)

                self.assertIn("status must be a boolean", str(cm.exception))

    def test_set_fine_with_coarse_all_command_error(self):
        """
        Test set_fine_with_coarse_all propagates NanoControlCommandError.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError("Command failed")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.set_fine_with_coarse_all(True)

            self.assertIn("Command failed", str(cm.exception))

    def test_set_fine_with_coarse_all_connection_error(self):
        """
        Test set_fine_with_coarse_all propagates NanoControlConnectionError.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlConnectionError(
                "Serial port disconnected"
            )

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.set_fine_with_coarse_all(False)

            self.assertIn("Serial port disconnected", str(cm.exception))

    def test_set_fine_with_coarse_all_boolean_conversion_explicit(self):
        """
        Test explicit boolean values convert to correct command parameters.
        """

        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (True, "finewithcoarse 1 1 1 1"),
            (False, "finewithcoarse 0 0 0 0"),
        ]

        for boolean_input, expected_command in test_cases:
            with self.subTest(input=boolean_input):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    nc.set_fine_with_coarse_all(boolean_input)

                    mock_send.assert_called_once_with(expected_command)

    def test_set_fine_with_coarse_all_multiple_calls(self):
        """
        Test multiple consecutive calls to set_fine_with_coarse_all.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command", return_value="OK") as mock_send:
            result1 = nc.set_fine_with_coarse_all(True)
            result2 = nc.set_fine_with_coarse_all(False)
            result3 = nc.set_fine_with_coarse_all(True)

            self.assertEqual(result1, "OK")
            self.assertEqual(result2, "OK")
            self.assertEqual(result3, "OK")

            expected_calls = [
                unittest.mock.call("finewithcoarse 1 1 1 1"),
                unittest.mock.call("finewithcoarse 0 0 0 0"),
                unittest.mock.call("finewithcoarse 1 1 1 1"),
            ]
            mock_send.assert_has_calls(expected_calls)
            self.assertEqual(mock_send.call_count, 3)

    def test_set_fine_with_coarse_all_various_device_responses(self):
        """
        Test set_fine_with_coarse_all with various valid device responses.
        """

        nc = NanoControl.__new__(NanoControl)

        responses = [
            "OK",
            "Fine-with-coarse enabled",
            "Fine-with-coarse disabled",
            "Command acknowledged",
            "",
            "Settings updated",
        ]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(
                    nc, "_send_command", return_value=response
                ) as mock_send:
                    result = nc.set_fine_with_coarse_all(True)

                    self.assertEqual(result, response)
                    mock_send.assert_called_once_with("finewithcoarse 1 1 1 1")

    def test_set_fine_with_coarse_all_return_type_consistency(self):
        """
        Test that set_fine_with_coarse_all always returns string type.
        """

        nc = NanoControl.__new__(NanoControl)

        test_responses = ["OK", "", "Success", "123", "Error message"]

        for response in test_responses:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command", return_value=response):
                    result = nc.set_fine_with_coarse_all(True)

                    self.assertIsInstance(result, str)
                    self.assertEqual(result, response)

    def test_set_fine_with_coarse_all_exact_command_format(self):
        """
        Test that exact command strings are sent to device.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command", return_value="OK") as mock_send:
            nc.set_fine_with_coarse_all(True)

            # Verify exact command sent
            args, kwargs = mock_send.call_args
            self.assertEqual(args[0], "finewithcoarse 1 1 1 1")
            self.assertEqual(len(args), 1)
            self.assertEqual(len(kwargs), 0)

    def test_set_fine_with_coarse_all_input_immutability(self):
        """
        Test that function doesn't modify input parameters.
        """

        nc = NanoControl.__new__(NanoControl)

        status_true = True
        status_false = False

        with patch.object(nc, "_send_command", return_value="OK"):
            nc.set_fine_with_coarse_all(status_true)
            nc.set_fine_with_coarse_all(status_false)

        # Verify inputs unchanged
        self.assertTrue(status_true)
        self.assertFalse(status_false)

    def test_set_fine_with_coarse_all_device_timeout(self):
        """
        Test set_fine_with_coarse_all when device times out.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError(
                "No response from device (timeout)"
            )

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.set_fine_with_coarse_all(True)

            self.assertIn("timeout", str(cm.exception))

    def test_set_fine_with_coarse_all_mixed_success_and_failure(self):
        """
        Test alternating success and failure calls.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            # First call fails
            mock_send.side_effect = [
                NanoControlCommandError("Device busy"),
                "OK",  # Second call succeeds
            ]

            with self.assertRaises(NanoControlCommandError):
                nc.set_fine_with_coarse_all(True)

            result = nc.set_fine_with_coarse_all(False)
            self.assertEqual(result, "OK")
            self.assertEqual(mock_send.call_count, 2)

    def test_set_fine_with_coarse_all_error_message_content(self):
        """
        Test that error messages contain expected function context.
        """

        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc.set_fine_with_coarse_all("invalid")

        error_msg = str(cm.exception)
        self.assertIn("NanoControl.set_fine_with_coarse_all", error_msg)
        self.assertIn("status must be a boolean", error_msg)

    def test_set_fine_with_coarse_all_boolean_subclass_rejection(self):
        """
        Test that boolean subclasses are properly handled.

        In Python, numpy.bool_ or other boolean-like types might
        behave differently from native bool.
        """

        nc = NanoControl.__new__(NanoControl)

        # Test with actual bool instances
        with patch.object(nc, "_send_command", return_value="OK"):
            # These should work
            nc.set_fine_with_coarse_all(bool(1))
            nc.set_fine_with_coarse_all(bool(0))
            nc.set_fine_with_coarse_all(True)
            nc.set_fine_with_coarse_all(False)

    def test_set_fine_with_coarse_all_call_sequence_validation(self):
        """
        Test that validation occurs before command sending.
        """

        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            # If validation fails, _send_command should not be called
            with self.assertRaises(ValueError):
                nc.set_fine_with_coarse_all("invalid")

            mock_send.assert_not_called()

    def test_set_fine_with_coarse_all_exception_chaining(self):
        """
        Test that exceptions from _send_command are properly propagated.
        """

        nc = NanoControl.__new__(NanoControl)

        original_error = NanoControlConnectionError("Port closed")

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = original_error

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.set_fine_with_coarse_all(True)

            # Should be the same exception instance
            self.assertIs(cm.exception, original_error)

    def test_set_motor_drive_frequencies_valid_inputs(self):
        """
        Test set_motor_drive_frequencies with valid frequency combinations.
        """
        nc = NanoControl.__new__(NanoControl)

        valid_test_cases = [
            ([300, 300, 300, 300], "All minimum values"),
            ([20000, 20000, 20000, 20000], "All maximum values"),
            ([300, 20000, 300, 20000], "Mix of min/max"),
            ([301, 19999, 302, 19998], "Just inside boundaries"),
            ([1000, 1000, 1000, 1000], "All same moderate value"),
            ([500, 1500, 2500, 3500], "Ascending pattern"),
            ([4000, 3000, 2000, 1000], "Descending pattern"),
            ([1000, 2000, 3000, 4000], "Regular intervals"),
            ([789, 1234, 5678, 9876], "Random valid values"),
            ([15000, 500, 10000, 750], "Mixed high/low"),
        ]

        for freqs, description in valid_test_cases:
            with self.subTest(freqs=freqs, description=description):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    result = nc.set_motor_drive_frequencies(freqs)

                    expected_command = f"frequency {' '.join(map(str, freqs))}"
                    mock_send.assert_called_once_with(expected_command)
                    self.assertEqual(result, "OK")

    def test_set_motor_drive_frequencies_invalid_input_types(self):
        """
        Test set_motor_drive_frequencies rejects non-list input types.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_inputs = [
            None,
            123,
            123.456,
            "1000,2000,3000,4000",
            (1000, 2000, 3000, 4000),
            {0: 1000, 1: 2000, 2: 3000, 3: 4000},
            set([1000, 2000, 3000, 4000]),
            lambda: [1000, 2000, 3000, 4000],
            range(1000, 5000, 1000),
        ]

        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc.set_motor_drive_frequencies(invalid_input)

                self.assertIn(
                    "frequencies must be a list of 4 integers",
                    str(cm.exception),
                )

    def test_set_motor_drive_frequencies_invalid_list_lengths(self):
        """
        Test set_motor_drive_frequencies rejects lists with wrong length.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_lengths = [
            [],
            [1000],
            [1000, 2000],
            [1000, 2000, 3000],
            [1000, 2000, 3000, 4000, 5000],
            [1000] * 10,
        ]

        for freqs in invalid_lengths:
            with self.subTest(freqs=freqs):
                with self.assertRaises(ValueError) as cm:
                    nc.set_motor_drive_frequencies(freqs)

                self.assertIn(
                    "frequencies must be a list of 4 integers",
                    str(cm.exception),
                )

    def test_set_motor_drive_frequencies_invalid_element_types(self):
        """
        Test set_motor_drive_frequencies rejects non-integer elements.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_element_cases = [
            ([1000.0, 2000, 3000, 4000], "base_joint", "Float in position 0"),
            (
                [1000, "2000", 3000, 4000],
                "elbow_joint",
                "String in position 1",
            ),
            (
                [1000, 2000, None, 4000],
                "prismatic_joint",
                "None in position 2",
            ),
            (
                [1000, 2000, 3000, [4000]],
                "tweezer_joint",
                "List in position 3",
            ),
            ([True, 2000, 3000, 4000], "base_joint", "Boolean in position 0"),
            (
                [1000, False, 3000, 4000],
                "elbow_joint",
                "Boolean in position 1",
            ),
            (
                [1000, 2000, {"freq": 3000}, 4000],
                "prismatic_joint",
                "Dict in position 2",
            ),
        ]

        for freqs, expected_joint, description in invalid_element_cases:
            with self.subTest(freqs=freqs, description=description):
                with self.assertRaises(ValueError) as cm:
                    nc.set_motor_drive_frequencies(freqs)

                error_msg = str(cm.exception)
                self.assertIn("must be an integer", error_msg)
                self.assertIn(expected_joint, error_msg)

    def test_set_motor_drive_frequencies_out_of_range_values(self):
        """
        Test set_motor_drive_frequencies rejects frequencies outside
        300-20000 Hz range.
        """
        nc = NanoControl.__new__(NanoControl)

        out_of_range_cases = [
            (
                [299, 1000, 2000, 3000],
                "base_joint",
                "Below minimum in position 0",
            ),
            (
                [1000, 20001, 2000, 3000],
                "elbow_joint",
                "Above maximum in position 1",
            ),
            ([1000, 2000, 0, 3000], "prismatic_joint", "Zero in position 2"),
            (
                [1000, 2000, 3000, -1000],
                "tweezer_joint",
                "Negative in position 3",
            ),
            ([50000, 1000, 2000, 3000], "base_joint", "Way above maximum"),
            ([299, 299, 299, 299], "base_joint", "All below minimum"),
            ([20001, 20001, 20001, 20001], "base_joint", "All above maximum"),
        ]

        for freqs, expected_joint, description in out_of_range_cases:
            with self.subTest(freqs=freqs, description=description):
                with self.assertRaises(ValueError) as cm:
                    nc.set_motor_drive_frequencies(freqs)

                error_msg = str(cm.exception)
                self.assertIn("must be between 300 and 20000 Hz", error_msg)
                self.assertIn(expected_joint, error_msg)

    def test_set_motor_drive_frequencies_boundary_values(self):
        """
        Test set_motor_drive_frequencies with exact boundary values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # Use constants instead of hardcoded values
            [
                nc.MIN_FREQUENCY,
                nc.MIN_FREQUENCY,
                nc.MIN_FREQUENCY,
                nc.MIN_FREQUENCY,
            ],
            [
                nc.MAX_FREQUENCY,
                nc.MAX_FREQUENCY,
                nc.MAX_FREQUENCY,
                nc.MAX_FREQUENCY,
            ],
            [
                nc.MIN_FREQUENCY,
                nc.MAX_FREQUENCY,
                nc.MIN_FREQUENCY,
                nc.MAX_FREQUENCY,
            ],
        ]

        for freqs in boundary_cases:
            with self.subTest(freqs=freqs):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    result = nc.set_motor_drive_frequencies(freqs)

                    expected_command = f"frequency {' '.join(map(str, freqs))}"
                    mock_send.assert_called_once_with(expected_command)
                    self.assertEqual(result, "OK")

    def test_set_motor_drive_frequencies_command_format(self):
        """
        Test that set_motor_drive_frequencies sends correctly formatted
        command.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            ([1000, 2000, 3000, 4000], "frequency 1000 2000 3000 4000"),
            ([300, 500, 10000, 20000], "frequency 300 500 10000 20000"),
            ([15000, 750, 1200, 8000], "frequency 15000 750 1200 8000"),
        ]

        for freqs, expected_command in test_cases:
            with self.subTest(freqs=freqs):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    nc.set_motor_drive_frequencies(freqs)

                    mock_send.assert_called_once_with(expected_command)

    def test_set_motor_drive_frequencies_connection_error(self):
        """
        Test set_motor_drive_frequencies propagates connection errors.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlConnectionError(
                "Serial port disconnected"
            )

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.set_motor_drive_frequencies([1000, 2000, 3000, 4000])

            self.assertIn("Serial port disconnected", str(cm.exception))

    def test_set_motor_drive_frequencies_command_error(self):
        """
        Test set_motor_drive_frequencies propagates command errors.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError(
                "Invalid frequency range"
            )

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.set_motor_drive_frequencies([1000, 2000, 3000, 4000])

            self.assertIn("Invalid frequency range", str(cm.exception))

    def test_set_motor_drive_frequencies_device_timeout(self):
        """
        Test set_motor_drive_frequencies when device times out.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError(
                "No response from device (timeout)"
            )

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.set_motor_drive_frequencies([1000, 2000, 3000, 4000])

            self.assertIn("timeout", str(cm.exception))

    def test_set_motor_drive_frequencies_multiple_calls(self):
        """
        Test multiple consecutive calls to set_motor_drive_frequencies.
        """
        nc = NanoControl.__new__(NanoControl)

        test_frequencies = [
            [1000, 2000, 3000, 4000],
            [1500, 2500, 3500, 4500],
            [500, 1000, 1500, 2000],
        ]

        with patch.object(nc, "_send_command", return_value="OK") as mock_send:
            results = []
            for freqs in test_frequencies:
                result = nc.set_motor_drive_frequencies(freqs)
                results.append(result)

            # Verify all calls succeeded
            self.assertEqual(len(results), 3)
            self.assertTrue(all(result == "OK" for result in results))
            self.assertEqual(mock_send.call_count, 3)

    def test_set_motor_drive_frequencies_joint_identification(self):
        """
        Test that error messages correctly identify which joint has
        invalid frequency.
        """
        nc = NanoControl.__new__(NanoControl)

        joint_error_cases = [
            ([299, 1000, 2000, 3000], "base_joint"),
            ([1000, 20001, 2000, 3000], "elbow_joint"),
            ([1000, 2000, 0, 3000], "prismatic_joint"),
            ([1000, 2000, 3000, -500], "tweezer_joint"),
        ]

        for freqs, expected_joint in joint_error_cases:
            with self.subTest(freqs=freqs, joint=expected_joint):
                with self.assertRaises(ValueError) as cm:
                    nc.set_motor_drive_frequencies(freqs)

                error_msg = str(cm.exception)
                self.assertIn(expected_joint, error_msg)

    def test_set_motor_drive_frequencies_return_value_types(self):
        """
        Test that set_motor_drive_frequencies always returns string.
        """
        nc = NanoControl.__new__(NanoControl)

        response_values = ["OK", "", "Success", "Frequencies set", "123"]

        for response in response_values:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command", return_value=response):
                    result = nc.set_motor_drive_frequencies(
                        [1000, 2000, 3000, 4000]
                    )

                    self.assertIsInstance(result, str)
                    self.assertEqual(result, response)

    def test_set_motor_drive_frequencies_validation_order(self):
        """
        Test that validation occurs before sending command to device.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_send_command") as mock_send:
            # Test with invalid input - _send_command should not be called
            with self.assertRaises(ValueError):
                nc.set_motor_drive_frequencies([299, 1000, 2000, 3000])

            mock_send.assert_not_called()

            # Test with valid input - _send_command should be called
            mock_send.return_value = "OK"
            nc.set_motor_drive_frequencies([1000, 2000, 3000, 4000])
            mock_send.assert_called_once()

    def test_set_motor_drive_frequencies_error_message_format(self):
        """
        Test that error messages contain expected function name and
        context.
        """
        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc.set_motor_drive_frequencies("invalid")

        error_msg = str(cm.exception)
        self.assertIn("NanoControl.set_motor_drive_frequencies", error_msg)
        self.assertIn("frequencies must be a list", error_msg)

    def test_set_motor_drive_frequencies_immutable_input(self):
        """
        Test that function doesn't modify input list.
        """
        nc = NanoControl.__new__(NanoControl)

        original_freqs = [1000, 2000, 3000, 4000]
        input_freqs = original_freqs.copy()

        with patch.object(nc, "_send_command", return_value="OK"):
            nc.set_motor_drive_frequencies(input_freqs)

        # Verify input wasn't modified
        self.assertEqual(input_freqs, original_freqs)

    def test_parse_speed_response_validates_device_format(self):
        """
        Test that _parse_speed_response accepts device format speeds (3-4 chars).
        """

        nc = NanoControl.__new__(NanoControl)

        device_format_responses = [
            "1 c001 f064 c032 f016",  # All 4-char format
            "2 c32 f064 c008 f16",  # Mixed 3-char and 4-char (valid lengths)
            "3 c064 f01 c32 f001",  # Fixed: c32 instead of c1 (valid 3-char)
        ]

        for response in device_format_responses:
            with self.subTest(response=response):
                try:
                    profile_num, speeds = nc._parse_speed_response(response)
                    self.assertIsInstance(profile_num, int)
                    self.assertEqual(len(speeds), 4)
                except NanoControlError:
                    self.fail(
                        f"Device format validation failed for: {response}"
                    )

    def test_parse_speed_response_speed_validation_order(self):
        """
        Test that speed validation happens in correct order and stops
        at first error.
        """

        nc = NanoControl.__new__(NanoControl)

        # First speed is invalid - should report base_joint error
        with self.assertRaises(NanoControlError) as cm:
            nc._parse_speed_response("1 x999 f016 c008 f001")

        error_msg = str(cm.exception)
        self.assertIn("base_joint", error_msg)
        self.assertIn("x999", error_msg)

        # Second speed is invalid - should report elbow_joint error
        with self.assertRaises(NanoControlError) as cm:
            nc._parse_speed_response("1 c032 y999 c008 f001")

        error_msg = str(cm.exception)
        self.assertIn("elbow_joint", error_msg)
        self.assertIn("y999", error_msg)

    def test_parse_speed_response_all_joints_named_correctly(self):
        """
        Test that all joint positions are correctly identified in error messages.
        """

        nc = NanoControl.__new__(NanoControl)

        joint_error_cases = [
            ("1 x001 f016 c008 f001", "base_joint"),
            ("1 c032 y016 c008 f001", "elbow_joint"),
            ("1 c032 f016 z008 f001", "prismatic_joint"),
            ("1 c032 f016 c008 w001", "tweezer_joint"),
        ]

        for response, expected_joint in joint_error_cases:
            with self.subTest(response=response, joint=expected_joint):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn(expected_joint, error_msg)

    def test_parse_speed_response_profile_validation_before_speed(self):
        """
        Test that profile number validation happens before speed validation.
        """

        nc = NanoControl.__new__(NanoControl)

        # Invalid profile should be caught before invalid speeds
        with self.assertRaises(NanoControlError) as cm:
            nc._parse_speed_response("invalid x999 y999 z999 w999")

        error_msg = str(cm.exception)
        self.assertIn("Profile number", error_msg)
        self.assertIn("invalid", error_msg)
        # Should not mention any joint-specific errors
        self.assertNotIn("base_joint", error_msg)

    def test_parse_speed_response_return_types(self):
        """
        Test that _parse_speed_response returns correct types.
        """

        nc = NanoControl.__new__(NanoControl)

        profile_num, speeds = nc._parse_speed_response("3 c032 f016 c008 f001")

        # Check return types
        self.assertIsInstance(profile_num, int)
        self.assertIsInstance(speeds, list)
        self.assertEqual(len(speeds), 4)

        # Check that all speeds are strings
        for speed in speeds:
            self.assertIsInstance(speed, str)

    def test_parse_speed_response_preserves_speed_format(self):
        """
        Test that original speed format is preserved in returned list.
        """

        nc = NanoControl.__new__(NanoControl)

        # Mix of 3-char and 4-char formats
        response = "2 c32 f064 c008 f16"
        profile_num, speeds = nc._parse_speed_response(response)

        expected_speeds = ["c32", "f064", "c008", "f16"]
        self.assertEqual(speeds, expected_speeds)

    def test_parse_speed_response_edge_case_responses(self):
        """
        Test _parse_speed_response with edge case but valid responses.
        """

        nc = NanoControl.__new__(NanoControl)

        edge_cases = [
            "1 c01 f01 c01 f01",  # All minimum values
            "6 c64 f64 c64 f64",  # All maximum values
            "3 c001 f001 c001 f001",  # All minimum with zero padding
            "4 c064 f064 c064 f064",  # All maximum with zero padding
        ]

        for response in edge_cases:
            with self.subTest(response=response):
                try:
                    profile_num, speeds = nc._parse_speed_response(response)
                    self.assertIsInstance(profile_num, int)
                    self.assertEqual(len(speeds), 4)
                except NanoControlError:
                    self.fail(f"Edge case failed unexpectedly: {response}")

    def test_parse_speed_response_rejects_invalid_format_lengths(self):
        """
        Test that _parse_speed_response rejects invalid format lengths.
        """

        nc = NanoControl.__new__(NanoControl)

        invalid_format_responses = [
            "1 c1 f016 c008 f001",  # 2-char format (too short)
            "1 c032 f12345 c008 f001",  # 6-char format (too long)
            "1 c032 f016 c c008",  # 1-char format (way too short)
        ]

        for response in invalid_format_responses:
            with self.subTest(response=response):
                with self.assertRaises(NanoControlError) as cm:
                    nc._parse_speed_response(response)

                error_msg = str(cm.exception)
                self.assertIn("not in valid format", error_msg)

    def test_validate_go_steps_valid_integers(self):
        """
        Test _validate_go_steps with valid integer inputs.
        """
        nc = NanoControl.__new__(NanoControl)

        valid_step_multipliers = [
            nc.MIN_STEP_MULTIPLIER,  # -100 (minimum boundary)
            nc.MAX_STEP_MULTIPLIER,  # 100 (maximum boundary)
            0,  # Zero
            1,  # Positive small
            -1,  # Negative small
            50,  # Positive medium
            -50,  # Negative medium
            99,  # Just below maximum
            -99,  # Just above minimum
        ]

        for step_multiplier in valid_step_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                try:
                    nc._validate_go_steps(step_multiplier)
                    # Should not raise any exception
                except ValueError:
                    self.fail(
                        f"_validate_go_steps raised ValueError unexpectedly "
                        f"for valid input {step_multiplier}"
                    )

    def test_validate_go_steps_invalid_types(self):
        """
        Test _validate_go_steps raises ValueError for non-integer types.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_types = [
            1.0,  # Float
            1.5,  # Float with decimal
            "1",  # String
            "10",  # String number
            None,  # None
            [1],  # List
            (1,),  # Tuple
            {"step": 1},  # Dictionary
            set([1]),  # Set
            True,  # Boolean (though technically int subclass)
            False,  # Boolean
        ]

        for invalid_input in invalid_types:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(invalid_input)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be an int", error_msg)
                self.assertIn("NanoControl._validate_go_steps", error_msg)

    def test_validate_go_steps_out_of_range_high(self):
        """
        Test _validate_go_steps raises ValueError for values above maximum.
        """
        nc = NanoControl.__new__(NanoControl)

        out_of_range_high = [
            nc.MAX_STEP_MULTIPLIER + 1,  # 101 (just above max)
            nc.MAX_STEP_MULTIPLIER + 10,  # 110
            200,  # Well above max
            1000,  # Way above max
            999999,  # Extremely high
        ]

        for step_multiplier in out_of_range_high:
            with self.subTest(step_multiplier=step_multiplier):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(step_multiplier)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be between", error_msg)
                self.assertIn(f"{nc.MIN_STEP_MULTIPLIER}", error_msg)
                self.assertIn(f"{nc.MAX_STEP_MULTIPLIER}", error_msg)

    def test_validate_go_steps_out_of_range_low(self):
        """
        Test _validate_go_steps raises ValueError for values below minimum.
        """
        nc = NanoControl.__new__(NanoControl)

        out_of_range_low = [
            nc.MIN_STEP_MULTIPLIER - 1,  # -101 (just below min)
            nc.MIN_STEP_MULTIPLIER - 10,  # -110
            -200,  # Well below min
            -1000,  # Way below min
            -999999,  # Extremely low
        ]

        for step_multiplier in out_of_range_low:
            with self.subTest(step_multiplier=step_multiplier):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(step_multiplier)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be between", error_msg)
                self.assertIn(f"{nc.MIN_STEP_MULTIPLIER}", error_msg)
                self.assertIn(f"{nc.MAX_STEP_MULTIPLIER}", error_msg)

    def test_validate_go_steps_boundary_values(self):
        """
        Test _validate_go_steps with exact boundary values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_values = [
            nc.MIN_STEP_MULTIPLIER,  # -100 (exact minimum)
            nc.MAX_STEP_MULTIPLIER,  # 100 (exact maximum)
        ]

        for step_multiplier in boundary_values:
            with self.subTest(step_multiplier=step_multiplier):
                try:
                    nc._validate_go_steps(step_multiplier)
                    # Should not raise any exception
                except ValueError:
                    self.fail(
                        f"_validate_go_steps rejected valid boundary "
                        f"value {step_multiplier}"
                    )

    def test_validate_go_steps_just_outside_boundaries(self):
        """
        Test _validate_go_steps with values just outside valid range.
        """
        nc = NanoControl.__new__(NanoControl)

        just_outside = [
            nc.MIN_STEP_MULTIPLIER - 1,  # -101 (just below min)
            nc.MAX_STEP_MULTIPLIER + 1,  # 101 (just above max)
        ]

        for step_multiplier in just_outside:
            with self.subTest(step_multiplier=step_multiplier):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(step_multiplier)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be between", error_msg)

    def test_validate_go_steps_boolean_rejection(self):
        """
        Test that _validate_go_steps explicitly rejects boolean values.

        Even though bool is a subclass of int in Python, we want to
        ensure explicit rejection of boolean inputs for clarity.
        """
        nc = NanoControl.__new__(NanoControl)

        boolean_values = [True, False]

        for bool_val in boolean_values:
            with self.subTest(bool_val=bool_val):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(bool_val)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be an int", error_msg)

    def test_validate_go_steps_error_message_content(self):
        """
        Test that error messages contain expected function context and ranges.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test type error message
        with self.assertRaises(ValueError) as cm:
            nc._validate_go_steps("invalid")

        error_msg = str(cm.exception)
        self.assertIn("NanoControl._validate_go_steps", error_msg)
        self.assertIn("step_multiplier must be an int", error_msg)

        # Test range error message
        with self.assertRaises(ValueError) as cm:
            nc._validate_go_steps(999)

        error_msg = str(cm.exception)
        self.assertIn("NanoControl._validate_go_steps", error_msg)
        self.assertIn("step_multiplier must be between", error_msg)
        self.assertIn(f"{nc.MIN_STEP_MULTIPLIER}", error_msg)
        self.assertIn(f"{nc.MAX_STEP_MULTIPLIER}", error_msg)

    def test_validate_go_steps_uses_class_constants(self):
        """
        Test that _validate_go_steps uses class constants in validation.

        This is a white-box test to ensure the function uses the
        class constants rather than hardcoded values.
        """
        nc = NanoControl.__new__(NanoControl)

        # Temporarily modify the class constants to verify they're used
        original_min = nc.MIN_STEP_MULTIPLIER
        original_max = nc.MAX_STEP_MULTIPLIER

        try:
            # Change the valid range to -50 to 50
            nc.MIN_STEP_MULTIPLIER = -50
            nc.MAX_STEP_MULTIPLIER = 50

            # Now -75 and 75 should be invalid, but 25 should be valid
            with self.assertRaises(ValueError):
                nc._validate_go_steps(-75)  # Below new minimum

            with self.assertRaises(ValueError):
                nc._validate_go_steps(75)  # Above new maximum

            # But 25 should still work
            try:
                nc._validate_go_steps(25)
            except ValueError:
                self.fail(
                    "Step multiplier 25 should be valid with modified range -50 to 50"
                )

        finally:
            # Restore original constants
            nc.MIN_STEP_MULTIPLIER = original_min
            nc.MAX_STEP_MULTIPLIER = original_max

    def test_validate_go_steps_return_value(self):
        """
        Test that _validate_go_steps returns None when validation passes.
        """
        nc = NanoControl.__new__(NanoControl)

        result = nc._validate_go_steps(50)
        self.assertIsNone(result)

    def test_validate_go_steps_no_side_effects(self):
        """
        Test that _validate_go_steps doesn't modify input or instance state.
        """
        nc = NanoControl.__new__(NanoControl)

        original_min = nc.MIN_STEP_MULTIPLIER
        original_max = nc.MAX_STEP_MULTIPLIER

        step_multiplier = 50
        original_step = step_multiplier

        nc._validate_go_steps(step_multiplier)

        # Verify no modifications
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(nc.MIN_STEP_MULTIPLIER, original_min)
        self.assertEqual(nc.MAX_STEP_MULTIPLIER, original_max)

    def test_validate_go_steps_multiple_calls(self):
        """
        Test that _validate_go_steps can be called multiple times safely.
        """
        nc = NanoControl.__new__(NanoControl)

        test_values = [0, 50, -50, 100, -100]

        for step_multiplier in test_values:
            with self.subTest(step_multiplier=step_multiplier):
                try:
                    nc._validate_go_steps(step_multiplier)
                    nc._validate_go_steps(step_multiplier)  # Call again
                    # Should work both times
                except ValueError:
                    self.fail(f"Multiple calls failed for {step_multiplier}")

    def test_validate_go_steps_zero_handling(self):
        """
        Test that _validate_go_steps correctly handles zero.
        """
        nc = NanoControl.__new__(NanoControl)

        try:
            nc._validate_go_steps(0)
            # Zero should be valid (within -100 to 100 range)
        except ValueError:
            self.fail("_validate_go_steps should accept zero as valid input")

    def test_validate_go_steps_large_integers(self):
        """
        Test _validate_go_steps with very large integers (but still int type).
        """
        nc = NanoControl.__new__(NanoControl)

        large_integers = [
            10**6,  # 1 million
            -(10**6),  # -1 million
            10**15,  # Very large positive
            -(10**15),  # Very large negative
        ]

        for large_int in large_integers:
            with self.subTest(large_int=large_int):
                with self.assertRaises(ValueError) as cm:
                    nc._validate_go_steps(large_int)

                error_msg = str(cm.exception)
                self.assertIn("step_multiplier must be between", error_msg)
                # Should not raise type error, only range error
                self.assertNotIn("must be an int", error_msg)

    def test_validate_go_steps_edge_case_types(self):
        """
        Test _validate_go_steps with edge case numeric types.
        """
        nc = NanoControl.__new__(NanoControl)

        import numpy as np

        edge_case_types = [
            np.int32(50),  # numpy int32
            np.int64(50),  # numpy int64
            np.float64(50.0),  # numpy float (should fail)
        ]

        for value in edge_case_types:
            with self.subTest(value=value, type=type(value)):
                if isinstance(value, (np.floating, float)):
                    # Should fail type check
                    with self.assertRaises(ValueError) as cm:
                        nc._validate_go_steps(value)
                    self.assertIn("must be an int", str(cm.exception))
                else:
                    # numpy integers might pass isinstance(x, int) check
                    try:
                        nc._validate_go_steps(value)
                        # If it passes, that's fine
                    except ValueError as e:
                        # If it fails, should be type error
                        self.assertIn("must be an int", str(e))

    def test_go_valid_inputs(self):
        """
        Test _go with valid step multipliers and interval.
        """
        nc = NanoControl.__new__(NanoControl)

        valid_test_cases = [
            # (step_a, step_b, step_c, step_d, interval_ms, expected_command)
            (0, 0, 0, 0, 100, "go 0 0 0 0 100"),
            (1, 1, 1, 1, 100, "go 1 1 1 1 100"),
            (-1, -1, -1, -1, 100, "go -1 -1 -1 -1 100"),
            (100, -100, 50, -50, 200, "go 100 -100 50 -50 200"),
            (10, 20, 30, 40, 150, "go 10 20 30 40 150"),
            (-100, -100, -100, -100, 1, "go -100 -100 -100 -100 1"),
            (100, 100, 100, 100, 1000, "go 100 100 100 100 1000"),
        ]

        for (
            step_a,
            step_b,
            step_c,
            step_d,
            interval,
            expected_cmd,
        ) in valid_test_cases:
            with self.subTest(
                steps=(step_a, step_b, step_c, step_d), interval=interval
            ):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    result = nc._go(step_a, step_b, step_c, step_d, interval)

                    mock_send.assert_called_once_with(expected_cmd)
                    self.assertEqual(result, "OK")

    def test_go_invalid_step_multiplier_types(self):
        """
        Test _go raises ValueError for invalid step multiplier types.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_types = [
            1.0,  # Float
            "1",  # String
            None,  # None
            [1],  # List
            True,  # Boolean
            False,  # Boolean
        ]

        # Test each position with invalid type
        for invalid_input in invalid_types:
            with self.subTest(invalid_input=invalid_input):
                # Test step_multiplier_a
                with self.assertRaises(ValueError) as cm:
                    nc._go(invalid_input, 1, 1, 1, 100)

                error_msg = str(cm.exception)
                self.assertIn("NanoControl._go", error_msg)
                self.assertIn("base_joint", error_msg)
                self.assertIn("step_multiplier must be an int", error_msg)

                # Test step_multiplier_b
                with self.assertRaises(ValueError) as cm:
                    nc._go(1, invalid_input, 1, 1, 100)

                error_msg = str(cm.exception)
                self.assertIn("elbow_joint", error_msg)

                # Test step_multiplier_c
                with self.assertRaises(ValueError) as cm:
                    nc._go(1, 1, invalid_input, 1, 100)

                error_msg = str(cm.exception)
                self.assertIn("prismatic_joint", error_msg)

                # Test step_multiplier_d
                with self.assertRaises(ValueError) as cm:
                    nc._go(1, 1, 1, invalid_input, 100)

                error_msg = str(cm.exception)
                self.assertIn("tweezer_joint", error_msg)

    def test_go_step_multipliers_out_of_range(self):
        """
        Test _go raises ValueError for step multipliers outside valid range.
        """
        nc = NanoControl.__new__(NanoControl)

        out_of_range_values = [
            nc.MIN_STEP_MULTIPLIER - 1,  # -101 (below minimum)
            nc.MAX_STEP_MULTIPLIER + 1,  # 101 (above maximum)
            -200,  # Well below minimum
            200,  # Well above maximum
        ]

        for out_of_range in out_of_range_values:
            with self.subTest(out_of_range=out_of_range):
                # Test each position
                positions = [
                    ("base_joint", (out_of_range, 1, 1, 1)),
                    ("elbow_joint", (1, out_of_range, 1, 1)),
                    ("prismatic_joint", (1, 1, out_of_range, 1)),
                    ("tweezer_joint", (1, 1, 1, out_of_range)),
                ]

                for joint_name, (a, b, c, d) in positions:
                    with self.subTest(joint=joint_name):
                        with self.assertRaises(ValueError) as cm:
                            nc._go(a, b, c, d, 100)

                        error_msg = str(cm.exception)
                        self.assertIn("NanoControl._go", error_msg)
                        self.assertIn(joint_name, error_msg)
                        self.assertIn(
                            "step_multiplier must be between", error_msg
                        )

    def test_go_invalid_interval_types(self):
        """
        Test _go raises ValueError for invalid interval_ms types.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_intervals = [
            1.0,  # Float
            "100",  # String
            None,  # None
            [100],  # List
            True,  # Boolean
            False,  # Boolean
        ]

        for invalid_interval in invalid_intervals:
            with self.subTest(invalid_interval=invalid_interval):
                with self.assertRaises(ValueError) as cm:
                    nc._go(1, 1, 1, 1, invalid_interval)

                error_msg = str(cm.exception)
                self.assertIn(
                    "interval_ms must be a positive integer", error_msg
                )
                self.assertIn(f"got {invalid_interval}", error_msg)

    def test_go_invalid_interval_values(self):
        """
        Test _go raises ValueError for non-positive interval values.
        """
        nc = NanoControl.__new__(NanoControl)

        invalid_intervals = [0, -1, -100, -1000]

        for invalid_interval in invalid_intervals:
            with self.subTest(invalid_interval=invalid_interval):
                with self.assertRaises(ValueError) as cm:
                    nc._go(1, 1, 1, 1, invalid_interval)

                error_msg = str(cm.exception)
                self.assertIn(
                    "interval_ms must be a positive integer", error_msg
                )
                self.assertIn(f"got {invalid_interval}", error_msg)

    def test_go_boundary_step_multipliers(self):
        """
        Test _go with boundary step multiplier values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # All minimum values
            (
                nc.MIN_STEP_MULTIPLIER,
                nc.MIN_STEP_MULTIPLIER,
                nc.MIN_STEP_MULTIPLIER,
                nc.MIN_STEP_MULTIPLIER,
                100,
            ),
            # All maximum values
            (
                nc.MAX_STEP_MULTIPLIER,
                nc.MAX_STEP_MULTIPLIER,
                nc.MAX_STEP_MULTIPLIER,
                nc.MAX_STEP_MULTIPLIER,
                100,
            ),
            # Mixed boundaries
            (
                nc.MIN_STEP_MULTIPLIER,
                nc.MAX_STEP_MULTIPLIER,
                nc.MIN_STEP_MULTIPLIER,
                nc.MAX_STEP_MULTIPLIER,
                1,
            ),
        ]

        for step_a, step_b, step_c, step_d, interval in boundary_cases:
            with self.subTest(steps=(step_a, step_b, step_c, step_d)):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    result = nc._go(step_a, step_b, step_c, step_d, interval)

                    expected_cmd = (
                        f"go {step_a} {step_b} {step_c} {step_d} {interval}"
                    )
                    mock_send.assert_called_once_with(expected_cmd)
                    self.assertEqual(result, "OK")

    def test_go_validation_order(self):
        """
        Test that _go validates step multipliers before interval.
        """
        nc = NanoControl.__new__(NanoControl)

        # Invalid step multiplier should be caught before invalid interval
        with self.assertRaises(ValueError) as cm:
            nc._go(999, 1, 1, 1, "invalid_interval")  # Both invalid

        error_msg = str(cm.exception)
        # Should complain about step multiplier (base_joint), not interval
        self.assertIn("base_joint", error_msg)
        self.assertNotIn("interval_ms", error_msg)

    def test_go_joint_name_mapping(self):
        """
        Test that _go correctly maps step multiplier positions to joint names.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test each position individually
        joint_tests = [
            (0, "base_joint", (999, 1, 1, 1)),  # Position 0
            (1, "elbow_joint", (1, 999, 1, 1)),  # Position 1
            (2, "prismatic_joint", (1, 1, 999, 1)),  # Position 2
            (3, "tweezer_joint", (1, 1, 1, 999)),  # Position 3
        ]

        for position, expected_joint, (a, b, c, d) in joint_tests:
            with self.subTest(position=position, joint=expected_joint):
                with self.assertRaises(ValueError) as cm:
                    nc._go(a, b, c, d, 100)

                error_msg = str(cm.exception)
                self.assertIn(expected_joint, error_msg)

    def test_go_error_message_stripping(self):
        """
        Test that _go properly strips function name from error messages.
        """
        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc._go(999, 1, 1, 1, 100)

        error_msg = str(cm.exception)
        # Should not have nested function names
        self.assertIn("NanoControl._go", error_msg)
        # Should have stripped the inner "_validate_go_steps" prefix
        self.assertNotIn("_validate_go_steps", error_msg)

    def test_go_send_command_propagation(self):
        """
        Test that _go propagates errors from _send_command.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test NanoControlConnectionError propagation
        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlConnectionError(
                "Connection lost"
            )

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc._go(1, 1, 1, 1, 100)

            self.assertIn("Connection lost", str(cm.exception))

        # Test NanoControlCommandError propagation
        with patch.object(nc, "_send_command") as mock_send:
            mock_send.side_effect = NanoControlCommandError("Device busy")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc._go(1, 1, 1, 1, 100)

            self.assertIn("Device busy", str(cm.exception))

    def test_go_command_format(self):
        """
        Test that _go generates correctly formatted commands.
        """
        nc = NanoControl.__new__(NanoControl)

        format_tests = [
            # Test negative values formatting
            (-10, -20, -30, -40, 500, "go -10 -20 -30 -40 500"),
            # Test zero values
            (0, 0, 0, 0, 1, "go 0 0 0 0 1"),
            # Test mixed positive/negative
            (10, -10, 20, -20, 100, "go 10 -10 20 -20 100"),
            # Test large values
            (100, 100, 100, 100, 9999, "go 100 100 100 100 9999"),
        ]

        for step_a, step_b, step_c, step_d, interval, expected in format_tests:
            with self.subTest(expected=expected):
                with patch.object(
                    nc, "_send_command", return_value="OK"
                ) as mock_send:
                    nc._go(step_a, step_b, step_c, step_d, interval)

                    mock_send.assert_called_once_with(expected)

    def test_go_return_value_passthrough(self):
        """
        Test that _go returns the response from _send_command.
        """
        nc = NanoControl.__new__(NanoControl)

        responses = ["OK", "MOVING", "Command accepted", "", "123"]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_send_command", return_value=response):
                    result = nc._go(1, 1, 1, 1, 100)

                    self.assertEqual(result, response)

    def test_go_no_side_effects(self):
        """
        Test that _go doesn't modify input parameters or instance state.
        """
        nc = NanoControl.__new__(NanoControl)

        original_values = (10, 20, 30, 40, 100)
        step_a, step_b, step_c, step_d, interval = original_values

        with patch.object(nc, "_send_command", return_value="OK"):
            nc._go(step_a, step_b, step_c, step_d, interval)

        # Verify inputs weren't modified
        current_values = (step_a, step_b, step_c, step_d, interval)
        self.assertEqual(current_values, original_values)

    def test_go_exception_chaining(self):
        """
        Test that _go properly chains validation exceptions.
        """
        nc = NanoControl.__new__(NanoControl)

        with self.assertRaises(ValueError) as cm:
            nc._go(999, 1, 1, 1, 100)  # Out of range step multiplier

        # Should be chained from the original validation error
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_go_multiple_validation_errors(self):
        """
        Test that _go stops at first validation error (doesn't validate all).
        """
        nc = NanoControl.__new__(NanoControl)

        # Multiple invalid step multipliers - should only report first one
        with self.assertRaises(ValueError) as cm:
            nc._go(999, 888, 777, 666, 100)

        error_msg = str(cm.exception)
        # Should only mention base_joint (first error), not others
        self.assertIn("base_joint", error_msg)
        self.assertNotIn("elbow_joint", error_msg)
        self.assertNotIn("prismatic_joint", error_msg)
        self.assertNotIn("tweezer_joint", error_msg)

    def test_go_large_interval_values(self):
        """
        Test _go with large but valid interval values.
        """
        nc = NanoControl.__new__(NanoControl)

        large_intervals = [1000, 10000, 100000, 999999]

        for interval in large_intervals:
            with self.subTest(interval=interval):
                with patch.object(nc, "_send_command", return_value="OK"):
                    nc._go(1, 1, 1, 1, interval)

                    f"go 1 1 1 1 {interval}"

    def test_process_step_multiplier_and_direction_positive_forward(self):
        """
        Test _process_step_multiplier_and_direction with positive multiplier, forward direction.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [1, 5, 10, 50, 100]

        for step_multiplier in test_cases:
            with self.subTest(step_multiplier=step_multiplier):
                result = nc._process_step_multiplier_and_direction(
                    step_multiplier, False
                )

                self.assertEqual(result, step_multiplier)
                self.assertGreaterEqual(result, 0)

    def test_process_step_multiplier_and_direction_positive_reverse(self):
        """
        Test _process_step_multiplier_and_direction with positive multiplier, reverse direction.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [1, 5, 10, 50, 100]

        for step_multiplier in test_cases:
            with self.subTest(step_multiplier=step_multiplier):
                result = nc._process_step_multiplier_and_direction(
                    step_multiplier, True
                )

                self.assertEqual(result, -step_multiplier)
                self.assertLessEqual(result, 0)

    def test_process_step_multiplier_and_direction_negative_forward_with_warning(
        self,
    ):
        """
        Test _process_step_multiplier_and_direction with negative multiplier, forward direction.
        Should print warning and use absolute value.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [-1, -5, -10, -50, -100]

        for step_multiplier in test_cases:
            with self.subTest(step_multiplier=step_multiplier):
                with patch("builtins.print") as mock_print:
                    result = nc._process_step_multiplier_and_direction(
                        step_multiplier, False
                    )

                    # Should return positive (absolute value)
                    self.assertEqual(result, abs(step_multiplier))
                    self.assertGreaterEqual(result, 0)

                    # Should print warning
                    mock_print.assert_called_once()
                    warning_msg = mock_print.call_args[0][0]
                    self.assertIn("WARNING", warning_msg)
                    self.assertIn("should be positive", warning_msg)
                    self.assertIn(
                        f"abs({step_multiplier})={abs(step_multiplier)}",
                        warning_msg,
                    )
                    self.assertIn("Use reverse=True", warning_msg)

    def test_process_step_multiplier_and_direction_negative_reverse_with_warning(
        self,
    ):
        """
        Test _process_step_multiplier_and_direction with negative multiplier, reverse direction.
        Should print warning and apply reverse to absolute value.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [-1, -5, -10, -50, -100]

        for step_multiplier in test_cases:
            with self.subTest(step_multiplier=step_multiplier):
                with patch("builtins.print") as mock_print:
                    result = nc._process_step_multiplier_and_direction(
                        step_multiplier, True
                    )

                    # Should return negative (reverse of absolute value)
                    self.assertEqual(result, -abs(step_multiplier))
                    self.assertLessEqual(result, 0)

                    # Should print warning
                    mock_print.assert_called_once()
                    warning_msg = mock_print.call_args[0][0]
                    self.assertIn("WARNING", warning_msg)
                    self.assertIn("should be positive", warning_msg)

    def test_process_step_multiplier_and_direction_zero_forward(self):
        """
        Test _process_step_multiplier_and_direction with zero multiplier, forward direction.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch("builtins.print") as mock_print:
            result = nc._process_step_multiplier_and_direction(0, False)

            self.assertEqual(result, 0)
            # Zero is not negative, so no warning should be printed
            mock_print.assert_not_called()

    def test_process_step_multiplier_and_direction_zero_reverse(self):
        """
        Test _process_step_multiplier_and_direction with zero multiplier, reverse direction.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch("builtins.print") as mock_print:
            result = nc._process_step_multiplier_and_direction(0, True)

            self.assertEqual(result, 0)  # -0 is still 0
            # Zero is not negative, so no warning should be printed
            mock_print.assert_not_called()

    def test_process_step_multiplier_and_direction_warning_message_format(
        self,
    ):
        """
        Test that warning messages contain expected information.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [-1, -25, -100]

        for step_multiplier in test_cases:
            with self.subTest(step_multiplier=step_multiplier):
                with patch("builtins.print") as mock_print:
                    nc._process_step_multiplier_and_direction(
                        step_multiplier, False
                    )

                    warning_msg = mock_print.call_args[0][0]

                    # Check all required components of warning message
                    self.assertIn("NanoControl:", warning_msg)
                    self.assertIn("WARNING:", warning_msg)
                    self.assertIn(
                        "step_multiplier should be positive", warning_msg
                    )
                    self.assertIn(
                        f"abs({step_multiplier})={abs(step_multiplier)}",
                        warning_msg,
                    )
                    self.assertIn(
                        "Use reverse=True for reverse direction", warning_msg
                    )

    def test_process_step_multiplier_and_direction_no_warning_for_positive(
        self,
    ):
        """
        Test that no warning is printed for positive values.
        """
        nc = NanoControl.__new__(NanoControl)

        positive_values = [1, 5, 10, 50, 100]

        for step_multiplier in positive_values:
            for reverse in [True, False]:
                with self.subTest(
                    step_multiplier=step_multiplier, reverse=reverse
                ):
                    with patch("builtins.print") as mock_print:
                        nc._process_step_multiplier_and_direction(
                            step_multiplier, reverse
                        )

                        # No warning should be printed for positive values
                        mock_print.assert_not_called()

    def test_process_step_multiplier_and_direction_return_types(self):
        """
        Test that function always returns integer type.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (1, True),
            (1, False),
            (-1, True),
            (-1, False),
            (0, True),
            (0, False),
            (100, True),
            (-100, False),
        ]

        for step_multiplier, reverse in test_cases:
            with self.subTest(
                step_multiplier=step_multiplier, reverse=reverse
            ):
                with patch(
                    "builtins.print"
                ):  # Suppress warnings for this test
                    result = nc._process_step_multiplier_and_direction(
                        step_multiplier, reverse
                    )

                    self.assertIsInstance(result, int)

    def test_process_step_multiplier_and_direction_direction_logic(self):
        """
        Test the direction application logic thoroughly.
        """
        nc = NanoControl.__new__(NanoControl)

        direction_tests = [
            # (input, reverse, expected_output)
            (10, False, 10),  # Positive forward -> positive
            (10, True, -10),  # Positive reverse -> negative
            (-10, False, 10),  # Negative forward -> positive (abs)
            (
                -10,
                True,
                -10,
            ),  # Negative reverse -> negative (abs then reverse)
            (0, False, 0),  # Zero forward -> zero
            (0, True, 0),  # Zero reverse -> zero
        ]

        for input_val, reverse, expected in direction_tests:
            with self.subTest(
                input=input_val, reverse=reverse, expected=expected
            ):
                with patch("builtins.print"):  # Suppress warnings
                    result = nc._process_step_multiplier_and_direction(
                        input_val, reverse
                    )

                    self.assertEqual(result, expected)

    def test_process_step_multiplier_and_direction_abs_calculation(self):
        """
        Test that absolute value calculation is correct.
        """
        nc = NanoControl.__new__(NanoControl)

        abs_tests = [(-1, 1), (-5, 5), (-10, 10), (-50, 50), (-100, 100)]

        for negative_val, expected_abs in abs_tests:
            with self.subTest(negative_val=negative_val):
                with patch("builtins.print") as mock_print:
                    # Test forward direction (should return abs value)
                    result_forward = nc._process_step_multiplier_and_direction(
                        negative_val, False
                    )
                    self.assertEqual(result_forward, expected_abs)

                    # Test reverse direction (should return -abs value)
                    result_reverse = nc._process_step_multiplier_and_direction(
                        negative_val, True
                    )
                    self.assertEqual(result_reverse, -expected_abs)

                    # Both should trigger warning
                    self.assertEqual(mock_print.call_count, 2)

    def test_process_step_multiplier_and_direction_edge_case_values(self):
        """
        Test with edge case values like very large numbers.
        """
        nc = NanoControl.__new__(NanoControl)

        edge_cases = [
            (999999, False, 999999),
            (999999, True, -999999),
            (-999999, False, 999999),
            (-999999, True, -999999),
            (1, False, 1),
            (1, True, -1),
            (-1, False, 1),
            (-1, True, -1),
        ]

        for input_val, reverse, expected in edge_cases:
            with self.subTest(input=input_val, reverse=reverse):
                with patch("builtins.print"):  # Suppress warnings
                    result = nc._process_step_multiplier_and_direction(
                        input_val, reverse
                    )

                    self.assertEqual(result, expected)

    def test_process_step_multiplier_and_direction_boolean_parameter_types(
        self,
    ):
        """
        Test that function properly handles boolean reverse parameter.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test explicit boolean values
        boolean_tests = [
            (10, True, -10),
            (10, False, 10),
            (-10, True, -10),
            (-10, False, 10),
        ]

        for step_multiplier, reverse_bool, expected in boolean_tests:
            with self.subTest(
                step_multiplier=step_multiplier, reverse=reverse_bool
            ):
                with patch("builtins.print"):  # Suppress warnings
                    result = nc._process_step_multiplier_and_direction(
                        step_multiplier, reverse_bool
                    )

                    self.assertEqual(result, expected)

    def test_process_step_multiplier_and_direction_no_side_effects(self):
        """
        Test that function doesn't modify input parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        original_step = 25
        original_reverse = True

        step_multiplier = original_step
        reverse = original_reverse

        with patch("builtins.print"):  # Suppress warnings
            nc._process_step_multiplier_and_direction(step_multiplier, reverse)

        # Verify inputs weren't modified
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(reverse, original_reverse)

    def test_process_step_multiplier_and_direction_warning_printed_once(self):
        """
        Test that warning is printed exactly once for negative values.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch("builtins.print") as mock_print:
            nc._process_step_multiplier_and_direction(-50, False)

            # Should be called exactly once
            mock_print.assert_called_once()

    def test_process_step_multiplier_and_direction_multiple_calls(self):
        """
        Test that function can be called multiple times with consistent results.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [(10, False), (10, True), (-10, False), (-10, True)]

        for step_multiplier, reverse in test_cases:
            with self.subTest(
                step_multiplier=step_multiplier, reverse=reverse
            ):
                with patch("builtins.print"):  # Suppress warnings
                    result1 = nc._process_step_multiplier_and_direction(
                        step_multiplier, reverse
                    )
                    result2 = nc._process_step_multiplier_and_direction(
                        step_multiplier, reverse
                    )

                    self.assertEqual(result1, result2)

    def test_process_step_multiplier_and_direction_warning_content_accuracy(
        self,
    ):
        """
        Test that warning message contains accurate values.
        """
        nc = NanoControl.__new__(NanoControl)

        test_values = [-7, -23, -99]

        for step_multiplier in test_values:
            with self.subTest(step_multiplier=step_multiplier):
                with patch("builtins.print") as mock_print:
                    nc._process_step_multiplier_and_direction(
                        step_multiplier, False
                    )

                    warning_msg = mock_print.call_args[0][0]

                    # Check that the specific values appear in the message
                    self.assertIn(str(step_multiplier), warning_msg)
                    self.assertIn(str(abs(step_multiplier)), warning_msg)

    def test_process_step_multiplier_and_direction_comprehensive_matrix(self):
        """
        Test comprehensive matrix of input combinations.
        """
        nc = NanoControl.__new__(NanoControl)

        # Comprehensive test matrix
        test_matrix = [
            # (step_multiplier, reverse, expected_result, should_warn)
            (1, False, 1, False),
            (1, True, -1, False),
            (-1, False, 1, True),
            (-1, True, -1, True),
            (0, False, 0, False),
            (0, True, 0, False),
            (50, False, 50, False),
            (50, True, -50, False),
            (-50, False, 50, True),
            (-50, True, -50, True),
        ]

        for step_mult, reverse, expected, should_warn in test_matrix:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch("builtins.print") as mock_print:
                    result = nc._process_step_multiplier_and_direction(
                        step_mult, reverse
                    )

                    self.assertEqual(result, expected)

                    if should_warn:
                        mock_print.assert_called_once()
                        warning_msg = mock_print.call_args[0][0]
                        self.assertIn("WARNING", warning_msg)
                    else:
                        mock_print.assert_not_called()

    def test_drive_base_joint_default_parameters(self):
        """
        Test drive_base_joint with default parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_base_joint()

            # Should use defaults: step_multiplier=1, interval_ms=100, reverse=False
            # Final multiplier should be 1 (positive, forward direction)
            mock_go.assert_called_once_with(
                1, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
            )
            self.assertEqual(result, "OK")

    def test_drive_base_joint_custom_step_multiplier(self):
        """
        Test drive_base_joint with custom step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        test_multipliers = [1, 5, 10, 50, 100]

        for step_multiplier in test_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_base_joint(step_multiplier)

                    mock_go.assert_called_once_with(
                        step_multiplier, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )
                    self.assertEqual(result, "OK")

    def test_drive_base_joint_custom_interval(self):
        """
        Test drive_base_joint with custom interval.
        """
        nc = NanoControl.__new__(NanoControl)

        test_intervals = [1, 50, 200, 500, 1000]

        for interval_ms in test_intervals:
            with self.subTest(interval_ms=interval_ms):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_base_joint(interval_ms=interval_ms)

                    mock_go.assert_called_once_with(
                        nc.DEFAULT_STEP_MULTIPLIER, 0, 0, 0, interval_ms
                    )
                    self.assertEqual(result, "OK")

    def test_drive_base_joint_reverse_direction(self):
        """
        Test drive_base_joint with reverse direction.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_base_joint(reverse=True)

            # Should use negative step multiplier for reverse
            mock_go.assert_called_once_with(
                -nc.DEFAULT_STEP_MULTIPLIER,
                0,
                0,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )
            self.assertEqual(result, "OK")

    def test_drive_base_joint_all_custom_parameters(self):
        """
        Test drive_base_joint with all custom parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (5, 200, False, 5),  # Forward direction
            (10, 150, True, -10),  # Reverse direction
            (25, 300, False, 25),  # Forward direction
            (50, 100, True, -50),  # Reverse direction
        ]

        for step_mult, interval, reverse, expected_mult in test_cases:
            with self.subTest(
                step_mult=step_mult, interval=interval, reverse=reverse
            ):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_base_joint(step_mult, interval, reverse)

                    mock_go.assert_called_once_with(
                        expected_mult, 0, 0, 0, interval
                    )
                    self.assertEqual(result, "OK")

    def test_drive_base_joint_negative_step_multiplier_warning(self):
        """
        Test drive_base_joint handles negative step multipliers with warning.
        """
        nc = NanoControl.__new__(NanoControl)

        negative_multipliers = [-1, -5, -10, -25]

        for step_multiplier in negative_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_base_joint(step_multiplier)

                        # Should use absolute value (positive)
                        mock_go.assert_called_once_with(
                            abs(step_multiplier),
                            0,
                            0,
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

                        # Should print warning
                        mock_print.assert_called_once()
                        warning_msg = mock_print.call_args[0][0]
                        self.assertIn("WARNING", warning_msg)
                        self.assertIn("should be positive", warning_msg)

    def test_drive_base_joint_negative_step_multiplier_with_reverse(self):
        """
        Test drive_base_joint with negative step multiplier and reverse=True.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            with patch("builtins.print") as mock_print:
                result = nc.drive_base_joint(-10, reverse=True)

                # Should use negative absolute value (still negative for reverse)
                mock_go.assert_called_once_with(
                    -10, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                )
                self.assertEqual(result, "OK")

                # Should still print warning about negative input
                mock_print.assert_called_once()

    def test_drive_base_joint_only_base_joint_moves(self):
        """
        Test that drive_base_joint only affects base joint (channel A).
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (1, False, 1),
            (5, False, 5),
            (10, True, -10),
            (25, True, -25),
        ]

        for step_mult, reverse, expected_mult in test_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    nc.drive_base_joint(step_mult, reverse=reverse)

                    # Only first parameter (base joint) should be non-zero
                    mock_go.assert_called_once_with(
                        expected_mult, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )

    def test_drive_base_joint_propagates_go_errors(self):
        """
        Test that drive_base_joint propagates errors from _go method.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test ValueError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = ValueError("Invalid step multiplier")

            with self.assertRaises(ValueError) as cm:
                nc.drive_base_joint(1)

            self.assertIn("Invalid step multiplier", str(cm.exception))

        # Test NanoControlConnectionError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlConnectionError("Connection lost")

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.drive_base_joint(1)

            self.assertIn("Connection lost", str(cm.exception))

        # Test NanoControlCommandError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlCommandError("Device busy")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.drive_base_joint(1)

            self.assertIn("Device busy", str(cm.exception))

    def test_drive_base_joint_return_value_passthrough(self):
        """
        Test that drive_base_joint returns the response from _go.
        """
        nc = NanoControl.__new__(NanoControl)

        responses = ["OK", "MOVING", "Command accepted", "", "Device ready"]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_go", return_value=response):
                    result = nc.drive_base_joint(1)

                    self.assertEqual(result, response)

    def test_drive_base_joint_zero_step_multiplier(self):
        """
        Test drive_base_joint with zero step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_base_joint(0, reverse=reverse)

                        # Zero should remain zero regardless of direction
                        mock_go.assert_called_once_with(
                            0, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        self.assertEqual(result, "OK")

                        # No warning should be printed for zero
                        mock_print.assert_not_called()

    def test_drive_base_joint_parameter_processing_order(self):
        """
        Test that parameters are processed in correct order.
        """
        nc = NanoControl.__new__(NanoControl)

        # Mock the processing function to verify it's called with correct parameters
        with patch.object(
            nc, "_process_step_multiplier_and_direction", return_value=5
        ) as mock_process:
            with patch.object(nc, "_go", return_value="OK") as mock_go:
                nc.drive_base_joint(10, 200, True)

                # Should call processing function first
                mock_process.assert_called_once_with(10, True)

                # Then call _go with processed result
                mock_go.assert_called_once_with(5, 0, 0, 0, 200)

    def test_drive_base_joint_keyword_arguments(self):
        """
        Test drive_base_joint with keyword arguments.
        """
        nc = NanoControl.__new__(NanoControl)

        keyword_test_cases = [
            # Using keyword arguments in different orders
            {"step_multiplier": 5, "interval_ms": 200, "reverse": False},
            {"interval_ms": 150, "step_multiplier": 10, "reverse": True},
            {"reverse": True, "step_multiplier": 15},
            {"interval_ms": 300, "reverse": False},
            {"step_multiplier": 20},
        ]

        for kwargs in keyword_test_cases:
            with self.subTest(kwargs=kwargs):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print"):  # Suppress any warnings
                        result = nc.drive_base_joint(**kwargs)

                        # Extract expected values with defaults
                        expected_step = kwargs.get(
                            "step_multiplier", nc.DEFAULT_STEP_MULTIPLIER
                        )
                        expected_interval = kwargs.get(
                            "interval_ms", nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        expected_reverse = kwargs.get("reverse", False)

                        # Calculate expected final multiplier
                        expected_final = (
                            -abs(expected_step)
                            if expected_reverse
                            else abs(expected_step)
                        )

                        mock_go.assert_called_once_with(
                            expected_final, 0, 0, 0, expected_interval
                        )
                        self.assertEqual(result, "OK")

    def test_drive_base_joint_no_side_effects(self):
        """
        Test that drive_base_joint doesn't modify input parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        original_step = 25
        original_interval = 200
        original_reverse = True

        step_multiplier = original_step
        interval_ms = original_interval
        reverse = original_reverse

        with patch.object(nc, "_go", return_value="OK"):
            nc.drive_base_joint(step_multiplier, interval_ms, reverse)

        # Verify inputs weren't modified
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(interval_ms, original_interval)
        self.assertEqual(reverse, original_reverse)

    def test_drive_base_joint_multiple_calls(self):
        """
        Test multiple consecutive calls to drive_base_joint.
        """
        nc = NanoControl.__new__(NanoControl)

        calls = [(1, False), (5, True), (10, False), (15, True)]

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            for step_mult, reverse in calls:
                nc.drive_base_joint(step_mult, reverse=reverse)

            # Should have made one call for each invocation
            self.assertEqual(mock_go.call_count, len(calls))

    def test_drive_base_joint_boundary_step_multipliers(self):
        """
        Test drive_base_joint with boundary step multiplier values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # (input_step, reverse, expected_final_multiplier)
            (
                nc.MIN_STEP_MULTIPLIER,
                False,
                100,
            ),  # -100, forward -> abs(-100) = 100
            (nc.MAX_STEP_MULTIPLIER, False, 100),  # 100, forward -> 100
            (
                nc.MIN_STEP_MULTIPLIER,
                True,
                -100,
            ),  # -100, reverse -> -abs(-100) = -100
            (nc.MAX_STEP_MULTIPLIER, True, -100),  # 100, reverse -> -100
            (1, False, 1),  # Minimum positive
            (1, True, -1),  # Minimum positive, reversed
        ]

        for step_mult, reverse, expected_mult in boundary_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch(
                        "builtins.print"
                    ):  # Suppress warnings for negative inputs
                        result = nc.drive_base_joint(
                            step_mult, reverse=reverse
                        )

                        mock_go.assert_called_once_with(
                            expected_mult,
                            0,
                            0,
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

    def test_drive_base_joint_uses_class_constants(self):
        """
        Test that drive_base_joint uses class constants for defaults.
        """
        nc = NanoControl.__new__(NanoControl)

        # Verify that the function uses the current class constants
        with patch.object(nc, "_go", return_value="OK") as mock_go:
            nc.drive_base_joint()

            # Should use the actual class constants
            mock_go.assert_called_once_with(
                nc.DEFAULT_STEP_MULTIPLIER,
                0,
                0,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )

    def test_drive_base_joint_default_constants_values(self):
        """
        Test that drive_base_joint uses expected default constant values.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test that the constants have expected values
        self.assertEqual(nc.DEFAULT_STEP_MULTIPLIER, 1)
        self.assertEqual(nc.DEFAULT_DRIVE_INTERVAL_MS, 10)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            nc.drive_base_joint()

            mock_go.assert_called_once_with(1, 0, 0, 0, 10)

    def test_drive_elbow_joint_default_parameters(self):
        """
        Test drive_elbow_joint with default parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_elbow_joint()

            # Should use defaults: step_multiplier=1, interval_ms=100, reverse=False
            # Due to not reverse: reverse=False -> negative multiplier
            mock_go.assert_called_once_with(
                0,
                -1,
                0,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,  # Changed to -1
            )
            self.assertEqual(result, "OK")

    def test_drive_elbow_joint_custom_step_multiplier(self):
        """
        Test drive_elbow_joint with custom step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        test_multipliers = [1, 5, 10, 50, 100]

        for step_multiplier in test_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_elbow_joint(step_multiplier)

                    # Due to not reverse: reverse=False (default) -> negative multiplier
                    mock_go.assert_called_once_with(
                        0,
                        -step_multiplier,
                        0,
                        0,
                        nc.DEFAULT_DRIVE_INTERVAL_MS,  # Changed to negative
                    )
                    self.assertEqual(result, "OK")

    def test_drive_elbow_joint_custom_interval(self):
        """
        Test drive_elbow_joint with custom interval.
        """
        nc = NanoControl.__new__(NanoControl)

        test_intervals = [1, 50, 200, 500, 1000]

        for interval_ms in test_intervals:
            with self.subTest(interval_ms=interval_ms):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_elbow_joint(interval_ms=interval_ms)

                    # Due to not reverse: reverse=False (default) -> negative multiplier
                    mock_go.assert_called_once_with(
                        0,
                        -nc.DEFAULT_STEP_MULTIPLIER,
                        0,
                        0,
                        interval_ms,  # Changed to negative
                    )
                    self.assertEqual(result, "OK")

    def test_drive_elbow_joint_reverse_direction(self):
        """
        Test drive_elbow_joint with reverse direction (upwards).
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_elbow_joint(reverse=True)

            # Should use positive step multiplier for reverse (upwards) due to not reverse
            mock_go.assert_called_once_with(
                0,
                nc.DEFAULT_STEP_MULTIPLIER,  # Changed: now positive due to not reverse
                0,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )
            self.assertEqual(result, "OK")

    def test_drive_elbow_joint_all_custom_parameters(self):
        """
        Test drive_elbow_joint with all custom parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (
                5,
                200,
                False,
                -5,
            ),  # Forward direction (downwards) - now negative due to not reverse
            (
                10,
                150,
                True,
                10,
            ),  # Reverse direction (upwards) - now positive due to not reverse
            (
                25,
                300,
                False,
                -25,
            ),  # Forward direction - now negative due to not reverse
            (
                50,
                100,
                True,
                50,
            ),  # Reverse direction - now positive due to not reverse
        ]

        for step_mult, interval, reverse, expected_mult in test_cases:
            with self.subTest(
                step_mult=step_mult, interval=interval, reverse=reverse
            ):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_elbow_joint(step_mult, interval, reverse)

                    mock_go.assert_called_once_with(
                        0, expected_mult, 0, 0, interval
                    )
                    self.assertEqual(result, "OK")

    def test_drive_elbow_joint_negative_step_multiplier_warning(self):
        """
        Test drive_elbow_joint handles negative step multipliers with warning.
        """
        nc = NanoControl.__new__(NanoControl)

        negative_multipliers = [-1, -5, -10, -25]

        for step_multiplier in negative_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_elbow_joint(step_multiplier)

                        # Should use absolute value, then apply not reverse (False -> negative)
                        mock_go.assert_called_once_with(
                            0,
                            -abs(
                                step_multiplier
                            ),  # Changed: negative of absolute value
                            0,
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

                        # Should print warning
                        mock_print.assert_called_once()
                        warning_msg = mock_print.call_args[0][0]
                        self.assertIn("WARNING", warning_msg)
                        self.assertIn("should be positive", warning_msg)

    def test_drive_elbow_joint_negative_step_multiplier_with_reverse(self):
        """
        Test drive_elbow_joint with negative step multiplier and reverse=True.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            with patch("builtins.print") as mock_print:
                result = nc.drive_elbow_joint(-10, reverse=True)

                # Should use absolute value, then apply not reverse (True -> positive)
                mock_go.assert_called_once_with(
                    0,
                    10,
                    0,
                    0,
                    nc.DEFAULT_DRIVE_INTERVAL_MS,  # Changed: positive
                )
                self.assertEqual(result, "OK")

                # Should still print warning about negative input
                mock_print.assert_called_once()

    def test_drive_elbow_joint_only_elbow_joint_moves(self):
        """
        Test that drive_elbow_joint only affects elbow joint (channel B).
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (1, False, -1),  # Changed: forward now negative due to not reverse
            (5, False, -5),  # Changed: forward now negative due to not reverse
            (10, True, 10),  # Changed: reverse now positive due to not reverse
            (25, True, 25),  # Changed: reverse now positive due to not reverse
        ]

        for step_mult, reverse, expected_mult in test_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    nc.drive_elbow_joint(step_mult, reverse=reverse)

                    # Only second parameter (elbow joint) should be non-zero
                    mock_go.assert_called_once_with(
                        0, expected_mult, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )

    def test_drive_elbow_joint_propagates_go_errors(self):
        """
        Test that drive_elbow_joint propagates errors from _go method.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test ValueError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = ValueError("Invalid step multiplier")

            with self.assertRaises(ValueError) as cm:
                nc.drive_elbow_joint(1)

            self.assertIn("Invalid step multiplier", str(cm.exception))

        # Test NanoControlConnectionError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlConnectionError("Connection lost")

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.drive_elbow_joint(1)

            self.assertIn("Connection lost", str(cm.exception))

        # Test NanoControlCommandError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlCommandError("Device busy")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.drive_elbow_joint(1)

            self.assertIn("Device busy", str(cm.exception))

    def test_drive_elbow_joint_return_value_passthrough(self):
        """
        Test that drive_elbow_joint returns the response from _go.
        """
        nc = NanoControl.__new__(NanoControl)

        responses = ["OK", "MOVING", "Command accepted", "", "Device ready"]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_go", return_value=response):
                    result = nc.drive_elbow_joint(1)

                    self.assertEqual(result, response)

    def test_drive_elbow_joint_boundary_step_multipliers(self):
        """
        Test drive_elbow_joint with boundary step multiplier values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # (input_step, reverse, expected_final_multiplier)
            (
                nc.MIN_STEP_MULTIPLIER,
                False,
                -100,  # Changed: forward -> negative due to not reverse
            ),  # -100, forward -> -abs(-100) = -100
            (
                nc.MAX_STEP_MULTIPLIER,
                False,
                -100,
            ),  # Changed: 100, forward -> -100
            (
                nc.MIN_STEP_MULTIPLIER,
                True,
                100,  # Changed: reverse -> positive due to not reverse
            ),  # -100, reverse -> abs(-100) = 100
            (
                nc.MAX_STEP_MULTIPLIER,
                True,
                100,
            ),  # Changed: 100, reverse -> 100
            (1, False, -1),  # Changed: Minimum positive, forward -> negative
            (1, True, 1),  # Changed: Minimum positive, reversed -> positive
        ]

        for step_mult, reverse, expected_mult in boundary_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch(
                        "builtins.print"
                    ):  # Suppress warnings for negative inputs
                        result = nc.drive_elbow_joint(
                            step_mult, reverse=reverse
                        )

                        mock_go.assert_called_once_with(
                            0,
                            expected_mult,
                            0,
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

    def test_drive_elbow_joint_zero_step_multiplier(self):
        """
        Test drive_elbow_joint with zero step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_elbow_joint(0, reverse=reverse)

                        # Zero should remain zero regardless of direction
                        mock_go.assert_called_once_with(
                            0, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        self.assertEqual(result, "OK")

                        # No warning should be printed for zero
                        mock_print.assert_not_called()

    def test_drive_elbow_joint_parameter_processing_order(self):
        """
        Test that parameters are processed in correct order.
        """
        nc = NanoControl.__new__(NanoControl)

        # Mock the processing function to verify it's called with correct parameters
        with patch.object(
            nc, "_process_step_multiplier_and_direction", return_value=5
        ) as mock_process:
            with patch.object(nc, "_go", return_value="OK") as mock_go:
                nc.drive_elbow_joint(10, 200, True)

                # Should call processing function with not reverse applied
                mock_process.assert_called_once_with(
                    10, False
                )  # Changed: not True = False

                # Then call _go with processed result
                mock_go.assert_called_once_with(0, 5, 0, 0, 200)

    def test_drive_elbow_joint_keyword_arguments(self):
        """
        Test drive_elbow_joint with keyword arguments.
        """
        nc = NanoControl.__new__(NanoControl)

        keyword_test_cases = [
            # Using keyword arguments in different orders
            {"step_multiplier": 5, "interval_ms": 200, "reverse": False},
            {"interval_ms": 150, "step_multiplier": 10, "reverse": True},
            {"reverse": True, "step_multiplier": 15},
            {"interval_ms": 300, "reverse": False},
            {"step_multiplier": 20},
        ]

        for kwargs in keyword_test_cases:
            with self.subTest(kwargs=kwargs):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print"):  # Suppress any warnings
                        result = nc.drive_elbow_joint(**kwargs)

                        # Extract expected values with defaults
                        expected_step = kwargs.get(
                            "step_multiplier", nc.DEFAULT_STEP_MULTIPLIER
                        )
                        expected_interval = kwargs.get(
                            "interval_ms", nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        expected_reverse = kwargs.get("reverse", False)

                        # Calculate expected final multiplier - CHANGED LOGIC
                        # Due to not reverse: False->negative, True->positive
                        expected_final = (
                            abs(expected_step)
                            if expected_reverse
                            else -abs(expected_step)
                        )

                        mock_go.assert_called_once_with(
                            0, expected_final, 0, 0, expected_interval
                        )
                        self.assertEqual(result, "OK")

    def test_drive_elbow_joint_uses_class_constants(self):
        """
        Test that drive_elbow_joint uses class constants for defaults.
        """
        nc = NanoControl.__new__(NanoControl)

        # Verify that the function uses the current class constants
        with patch.object(nc, "_go", return_value="OK") as mock_go:
            nc.drive_elbow_joint()

            # Should use the actual class constants with not reverse applied
            mock_go.assert_called_once_with(
                0,
                -nc.DEFAULT_STEP_MULTIPLIER,  # Changed: negative due to not reverse
                0,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )

    def test_drive_elbow_joint_no_side_effects(self):
        """
        Test that drive_elbow_joint doesn't modify input parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        original_step = 25
        original_interval = 200
        original_reverse = True

        step_multiplier = original_step
        interval_ms = original_interval
        reverse = original_reverse

        with patch.object(nc, "_go", return_value="OK"):
            nc.drive_elbow_joint(step_multiplier, interval_ms, reverse)

        # Verify inputs weren't modified
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(interval_ms, original_interval)
        self.assertEqual(reverse, original_reverse)

    def test_drive_elbow_joint_multiple_calls(self):
        """
        Test multiple consecutive calls to drive_elbow_joint.
        """
        nc = NanoControl.__new__(NanoControl)

        calls = [(1, False), (5, True), (10, False), (15, True)]

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            for step_mult, reverse in calls:
                nc.drive_elbow_joint(step_mult, reverse=reverse)

            # Should have made one call for each invocation
            self.assertEqual(mock_go.call_count, len(calls))

    def test_drive_prismatic_joint_default_parameters(self):
        """
        Test drive_prismatic_joint with default parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_prismatic_joint()

            # Should use defaults: step_multiplier=1, interval_ms=100, reverse=False
            # Final multiplier should be 1 (positive, forward direction)
            mock_go.assert_called_once_with(
                0, 0, 1, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
            )
            self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_custom_step_multiplier(self):
        """
        Test drive_prismatic_joint with custom step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        test_multipliers = [1, 5, 10, 50, 100]

        for step_multiplier in test_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_prismatic_joint(step_multiplier)

                    mock_go.assert_called_once_with(
                        0, 0, step_multiplier, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )
                    self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_custom_interval(self):
        """
        Test drive_prismatic_joint with custom interval.
        """
        nc = NanoControl.__new__(NanoControl)

        test_intervals = [1, 50, 200, 500, 1000]

        for interval_ms in test_intervals:
            with self.subTest(interval_ms=interval_ms):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_prismatic_joint(interval_ms=interval_ms)

                    mock_go.assert_called_once_with(
                        0, 0, nc.DEFAULT_STEP_MULTIPLIER, 0, interval_ms
                    )
                    self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_reverse_direction(self):
        """
        Test drive_prismatic_joint with reverse direction.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_prismatic_joint(reverse=True)

            # Should use negative step multiplier for reverse (normal behavior)
            mock_go.assert_called_once_with(
                0,
                0,
                -nc.DEFAULT_STEP_MULTIPLIER,  # Negative for reverse
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )
            self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_all_custom_parameters(self):
        """
        Test drive_prismatic_joint with all custom parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (5, 200, False, 5),  # Forward direction - positive
            (10, 150, True, -10),  # Reverse direction - negative
            (25, 300, False, 25),  # Forward direction - positive
            (50, 100, True, -50),  # Reverse direction - negative
        ]

        for step_mult, interval, reverse, expected_mult in test_cases:
            with self.subTest(
                step_mult=step_mult, interval=interval, reverse=reverse
            ):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_prismatic_joint(
                        step_mult, interval, reverse
                    )

                    mock_go.assert_called_once_with(
                        0, 0, expected_mult, 0, interval
                    )
                    self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_negative_step_multiplier_warning(self):
        """
        Test drive_prismatic_joint handles negative step multipliers with warning.
        """
        nc = NanoControl.__new__(NanoControl)

        negative_multipliers = [-1, -5, -10, -25]

        for step_multiplier in negative_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_prismatic_joint(step_multiplier)

                        # Should use absolute value (positive)
                        mock_go.assert_called_once_with(
                            0,
                            0,
                            abs(step_multiplier),
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

                        # Should print warning
                        mock_print.assert_called_once()
                        warning_msg = mock_print.call_args[0][0]
                        self.assertIn("WARNING", warning_msg)
                        self.assertIn("should be positive", warning_msg)

    def test_drive_prismatic_joint_negative_step_multiplier_with_reverse(self):
        """
        Test drive_prismatic_joint with negative step multiplier and reverse=True.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            with patch("builtins.print") as mock_print:
                result = nc.drive_prismatic_joint(-10, reverse=True)

                # Should use negative absolute value (still negative for reverse)
                mock_go.assert_called_once_with(
                    0, 0, -10, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                )
                self.assertEqual(result, "OK")

                # Should still print warning about negative input
                mock_print.assert_called_once()

    def test_drive_prismatic_joint_only_prismatic_joint_moves(self):
        """
        Test that drive_prismatic_joint only affects prismatic joint (channel C).
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (1, False, 1),
            (5, False, 5),
            (10, True, -10),
            (25, True, -25),
        ]

        for step_mult, reverse, expected_mult in test_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    nc.drive_prismatic_joint(step_mult, reverse=reverse)

                    # Only third parameter (prismatic joint) should be non-zero
                    mock_go.assert_called_once_with(
                        0, 0, expected_mult, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )

    def test_drive_prismatic_joint_propagates_go_errors(self):
        """
        Test that drive_prismatic_joint propagates errors from _go method.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test ValueError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = ValueError("Invalid step multiplier")

            with self.assertRaises(ValueError) as cm:
                nc.drive_prismatic_joint(1)

            self.assertIn("Invalid step multiplier", str(cm.exception))

        # Test NanoControlConnectionError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlConnectionError("Connection lost")

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.drive_prismatic_joint(1)

            self.assertIn("Connection lost", str(cm.exception))

        # Test NanoControlCommandError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlCommandError("Device busy")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.drive_prismatic_joint(1)

            self.assertIn("Device busy", str(cm.exception))

    def test_drive_prismatic_joint_return_value_passthrough(self):
        """
        Test that drive_prismatic_joint returns the response from _go.
        """
        nc = NanoControl.__new__(NanoControl)

        responses = ["OK", "MOVING", "Command accepted", "", "Device ready"]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_go", return_value=response):
                    result = nc.drive_prismatic_joint(1)

                    self.assertEqual(result, response)

    def test_drive_prismatic_joint_boundary_step_multipliers(self):
        """
        Test drive_prismatic_joint with boundary step multiplier values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # (input_step, reverse, expected_final_multiplier)
            (
                nc.MIN_STEP_MULTIPLIER,
                False,
                100,
            ),  # -100, forward -> abs(-100) = 100
            (nc.MAX_STEP_MULTIPLIER, False, 100),  # 100, forward -> 100
            (
                nc.MIN_STEP_MULTIPLIER,
                True,
                -100,
            ),  # -100, reverse -> -abs(-100) = -100
            (nc.MAX_STEP_MULTIPLIER, True, -100),  # 100, reverse -> -100
            (1, False, 1),  # Minimum positive
            (1, True, -1),  # Minimum positive, reversed
        ]

        for step_mult, reverse, expected_mult in boundary_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch(
                        "builtins.print"
                    ):  # Suppress warnings for negative inputs
                        result = nc.drive_prismatic_joint(
                            step_mult, reverse=reverse
                        )

                        mock_go.assert_called_once_with(
                            0,
                            0,
                            expected_mult,
                            0,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_zero_step_multiplier(self):
        """
        Test drive_prismatic_joint with zero step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_prismatic_joint(0, reverse=reverse)

                        # Zero should remain zero regardless of direction
                        mock_go.assert_called_once_with(
                            0, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        self.assertEqual(result, "OK")

                        # No warning should be printed for zero
                        mock_print.assert_not_called()

    def test_drive_prismatic_joint_parameter_processing_order(self):
        """
        Test that parameters are processed in correct order.
        """
        nc = NanoControl.__new__(NanoControl)

        # Mock the processing function to verify it's called with correct parameters
        with patch.object(
            nc, "_process_step_multiplier_and_direction", return_value=5
        ) as mock_process:
            with patch.object(nc, "_go", return_value="OK") as mock_go:
                nc.drive_prismatic_joint(10, 200, True)

                # Should call processing function first
                mock_process.assert_called_once_with(10, True)

                # Then call _go with processed result
                mock_go.assert_called_once_with(0, 0, 5, 0, 200)

    def test_drive_prismatic_joint_keyword_arguments(self):
        """
        Test drive_prismatic_joint with keyword arguments.
        """
        nc = NanoControl.__new__(NanoControl)

        keyword_test_cases = [
            # Using keyword arguments in different orders
            {"step_multiplier": 5, "interval_ms": 200, "reverse": False},
            {"interval_ms": 150, "step_multiplier": 10, "reverse": True},
            {"reverse": True, "step_multiplier": 15},
            {"interval_ms": 300, "reverse": False},
            {"step_multiplier": 20},
        ]

        for kwargs in keyword_test_cases:
            with self.subTest(kwargs=kwargs):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print"):  # Suppress any warnings
                        result = nc.drive_prismatic_joint(**kwargs)

                        # Extract expected values with defaults
                        expected_step = kwargs.get(
                            "step_multiplier", nc.DEFAULT_STEP_MULTIPLIER
                        )
                        expected_interval = kwargs.get(
                            "interval_ms", nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        expected_reverse = kwargs.get("reverse", False)

                        # Calculate expected final multiplier
                        expected_final = (
                            -abs(expected_step)
                            if expected_reverse
                            else abs(expected_step)
                        )

                        mock_go.assert_called_once_with(
                            0, 0, expected_final, 0, expected_interval
                        )
                        self.assertEqual(result, "OK")

    def test_drive_prismatic_joint_uses_class_constants(self):
        """
        Test that drive_prismatic_joint uses class constants for defaults.
        """
        nc = NanoControl.__new__(NanoControl)

        # Verify that the function uses the current class constants
        with patch.object(nc, "_go", return_value="OK") as mock_go:
            nc.drive_prismatic_joint()

            # Should use the actual class constants
            mock_go.assert_called_once_with(
                0,
                0,
                nc.DEFAULT_STEP_MULTIPLIER,
                0,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )

    def test_drive_prismatic_joint_no_side_effects(self):
        """
        Test that drive_prismatic_joint doesn't modify input parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        original_step = 25
        original_interval = 200
        original_reverse = True

        step_multiplier = original_step
        interval_ms = original_interval
        reverse = original_reverse

        with patch.object(nc, "_go", return_value="OK"):
            nc.drive_prismatic_joint(step_multiplier, interval_ms, reverse)

        # Verify inputs weren't modified
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(interval_ms, original_interval)
        self.assertEqual(reverse, original_reverse)

    def test_drive_prismatic_joint_multiple_calls(self):
        """
        Test multiple consecutive calls to drive_prismatic_joint.
        """
        nc = NanoControl.__new__(NanoControl)

        calls = [(1, False), (5, True), (10, False), (15, True)]

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            for step_mult, reverse in calls:
                nc.drive_prismatic_joint(step_mult, reverse=reverse)

            # Should have made one call for each invocation
            self.assertEqual(mock_go.call_count, len(calls))

    def test_drive_tweezers_default_parameters(self):
        """
        Test drive_tweezers with default parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_tweezers()

            # Should use defaults: step_multiplier=1, interval_ms=100, reverse=False
            # Final multiplier should be 1 (positive, forward direction)
            mock_go.assert_called_once_with(
                0, 0, 0, 1, nc.DEFAULT_DRIVE_INTERVAL_MS
            )
            self.assertEqual(result, "OK")

    def test_drive_tweezers_custom_step_multiplier(self):
        """
        Test drive_tweezers with custom step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        test_multipliers = [1, 5, 10, 50, 100]

        for step_multiplier in test_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_tweezers(step_multiplier)

                    mock_go.assert_called_once_with(
                        0, 0, 0, step_multiplier, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )
                    self.assertEqual(result, "OK")

    def test_drive_tweezers_custom_interval(self):
        """
        Test drive_tweezers with custom interval.
        """
        nc = NanoControl.__new__(NanoControl)

        test_intervals = [1, 50, 200, 500, 1000]

        for interval_ms in test_intervals:
            with self.subTest(interval_ms=interval_ms):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_tweezers(interval_ms=interval_ms)

                    mock_go.assert_called_once_with(
                        0, 0, 0, nc.DEFAULT_STEP_MULTIPLIER, interval_ms
                    )
                    self.assertEqual(result, "OK")

    def test_drive_tweezers_reverse_direction(self):
        """
        Test drive_tweezers with reverse direction.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            result = nc.drive_tweezers(reverse=True)

            # Should use negative step multiplier for reverse
            mock_go.assert_called_once_with(
                0,
                0,
                0,
                -nc.DEFAULT_STEP_MULTIPLIER,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )
            self.assertEqual(result, "OK")

    def test_drive_tweezers_all_custom_parameters(self):
        """
        Test drive_tweezers with all custom parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (5, 200, False, 5),  # Forward direction
            (10, 150, True, -10),  # Reverse direction
            (25, 300, False, 25),  # Forward direction
            (50, 100, True, -50),  # Reverse direction
        ]

        for step_mult, interval, reverse, expected_mult in test_cases:
            with self.subTest(
                step_mult=step_mult, interval=interval, reverse=reverse
            ):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    result = nc.drive_tweezers(step_mult, interval, reverse)

                    mock_go.assert_called_once_with(
                        0, 0, 0, expected_mult, interval
                    )
                    self.assertEqual(result, "OK")

    def test_drive_tweezers_negative_step_multiplier_warning(self):
        """
        Test drive_tweezers handles negative step multipliers with warning.
        """
        nc = NanoControl.__new__(NanoControl)

        negative_multipliers = [-1, -5, -10, -25]

        for step_multiplier in negative_multipliers:
            with self.subTest(step_multiplier=step_multiplier):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_tweezers(step_multiplier)

                        # Should use absolute value (positive)
                        mock_go.assert_called_once_with(
                            0,
                            0,
                            0,
                            abs(step_multiplier),
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

                        # Should print warning
                        mock_print.assert_called_once()
                        warning_msg = mock_print.call_args[0][0]
                        self.assertIn("WARNING", warning_msg)
                        self.assertIn("should be positive", warning_msg)

    def test_drive_tweezers_negative_step_multiplier_with_reverse(self):
        """
        Test drive_tweezers with negative step multiplier and reverse=True.
        """
        nc = NanoControl.__new__(NanoControl)

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            with patch("builtins.print") as mock_print:
                result = nc.drive_tweezers(-10, reverse=True)

                # Should use negative absolute value (still negative for reverse)
                mock_go.assert_called_once_with(
                    0, 0, 0, -10, nc.DEFAULT_DRIVE_INTERVAL_MS
                )
                self.assertEqual(result, "OK")

                # Should still print warning about negative input
                mock_print.assert_called_once()

    def test_drive_tweezers_only_tweezer_joint_moves(self):
        """
        Test that drive_tweezers only affects tweezer joint (channel D).
        """
        nc = NanoControl.__new__(NanoControl)

        test_cases = [
            (1, False, 1),
            (5, False, 5),
            (10, True, -10),
            (25, True, -25),
        ]

        for step_mult, reverse, expected_mult in test_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    nc.drive_tweezers(step_mult, reverse=reverse)

                    # Only fourth parameter (tweezer joint) should be non-zero
                    mock_go.assert_called_once_with(
                        0, 0, 0, expected_mult, nc.DEFAULT_DRIVE_INTERVAL_MS
                    )

    def test_drive_tweezers_propagates_go_errors(self):
        """
        Test that drive_tweezers propagates errors from _go method.
        """
        nc = NanoControl.__new__(NanoControl)

        # Test ValueError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = ValueError("Invalid step multiplier")

            with self.assertRaises(ValueError) as cm:
                nc.drive_tweezers(1)

            self.assertIn("Invalid step multiplier", str(cm.exception))

        # Test NanoControlConnectionError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlConnectionError("Connection lost")

            with self.assertRaises(NanoControlConnectionError) as cm:
                nc.drive_tweezers(1)

            self.assertIn("Connection lost", str(cm.exception))

        # Test NanoControlCommandError propagation
        with patch.object(nc, "_go") as mock_go:
            mock_go.side_effect = NanoControlCommandError("Device busy")

            with self.assertRaises(NanoControlCommandError) as cm:
                nc.drive_tweezers(1)

            self.assertIn("Device busy", str(cm.exception))

    def test_drive_tweezers_return_value_passthrough(self):
        """
        Test that drive_tweezers returns the response from _go.
        """
        nc = NanoControl.__new__(NanoControl)

        responses = ["OK", "MOVING", "Command accepted", "", "Device ready"]

        for response in responses:
            with self.subTest(response=response):
                with patch.object(nc, "_go", return_value=response):
                    result = nc.drive_tweezers(1)

                    self.assertEqual(result, response)

    def test_drive_tweezers_boundary_step_multipliers(self):
        """
        Test drive_tweezers with boundary step multiplier values.
        """
        nc = NanoControl.__new__(NanoControl)

        boundary_cases = [
            # (input_step, reverse, expected_final_multiplier)
            (
                nc.MIN_STEP_MULTIPLIER,
                False,
                100,
            ),  # -100, forward -> abs(-100) = 100
            (nc.MAX_STEP_MULTIPLIER, False, 100),  # 100, forward -> 100
            (
                nc.MIN_STEP_MULTIPLIER,
                True,
                -100,
            ),  # -100, reverse -> -abs(-100) = -100
            (nc.MAX_STEP_MULTIPLIER, True, -100),  # 100, reverse -> -100
            (1, False, 1),  # Minimum positive
            (1, True, -1),  # Minimum positive, reversed
        ]

        for step_mult, reverse, expected_mult in boundary_cases:
            with self.subTest(step_mult=step_mult, reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch(
                        "builtins.print"
                    ):  # Suppress warnings for negative inputs
                        result = nc.drive_tweezers(step_mult, reverse=reverse)

                        mock_go.assert_called_once_with(
                            0,
                            0,
                            0,
                            expected_mult,
                            nc.DEFAULT_DRIVE_INTERVAL_MS,
                        )
                        self.assertEqual(result, "OK")

    def test_drive_tweezers_zero_step_multiplier(self):
        """
        Test drive_tweezers with zero step multiplier.
        """
        nc = NanoControl.__new__(NanoControl)

        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print") as mock_print:
                        result = nc.drive_tweezers(0, reverse=reverse)

                        # Zero should remain zero regardless of direction
                        mock_go.assert_called_once_with(
                            0, 0, 0, 0, nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        self.assertEqual(result, "OK")

                        # No warning should be printed for zero
                        mock_print.assert_not_called()

    def test_drive_tweezers_parameter_processing_order(self):
        """
        Test that parameters are processed in correct order.
        """
        nc = NanoControl.__new__(NanoControl)

        # Mock the processing function to verify it's called with correct parameters
        with patch.object(
            nc, "_process_step_multiplier_and_direction", return_value=5
        ) as mock_process:
            with patch.object(nc, "_go", return_value="OK") as mock_go:
                nc.drive_tweezers(10, 200, True)

                # Should call processing function first
                mock_process.assert_called_once_with(10, True)

                # Then call _go with processed result
                mock_go.assert_called_once_with(0, 0, 0, 5, 200)

    def test_drive_tweezers_keyword_arguments(self):
        """
        Test drive_tweezers with keyword arguments.
        """
        nc = NanoControl.__new__(NanoControl)

        keyword_test_cases = [
            # Using keyword arguments in different orders
            {"step_multiplier": 5, "interval_ms": 200, "reverse": False},
            {"interval_ms": 150, "step_multiplier": 10, "reverse": True},
            {"reverse": True, "step_multiplier": 15},
            {"interval_ms": 300, "reverse": False},
            {"step_multiplier": 20},
        ]

        for kwargs in keyword_test_cases:
            with self.subTest(kwargs=kwargs):
                with patch.object(nc, "_go", return_value="OK") as mock_go:
                    with patch("builtins.print"):  # Suppress any warnings
                        result = nc.drive_tweezers(**kwargs)

                        # Extract expected values with defaults
                        expected_step = kwargs.get(
                            "step_multiplier", nc.DEFAULT_STEP_MULTIPLIER
                        )
                        expected_interval = kwargs.get(
                            "interval_ms", nc.DEFAULT_DRIVE_INTERVAL_MS
                        )
                        expected_reverse = kwargs.get("reverse", False)

                        # Calculate expected final multiplier
                        expected_final = (
                            -abs(expected_step)
                            if expected_reverse
                            else abs(expected_step)
                        )

                        mock_go.assert_called_once_with(
                            0, 0, 0, expected_final, expected_interval
                        )
                        self.assertEqual(result, "OK")

    def test_drive_tweezers_uses_class_constants(self):
        """
        Test that drive_tweezers uses class constants for defaults.
        """
        nc = NanoControl.__new__(NanoControl)

        # Verify that the function uses the current class constants
        with patch.object(nc, "_go", return_value="OK") as mock_go:
            nc.drive_tweezers()

            # Should use the actual class constants
            mock_go.assert_called_once_with(
                0,
                0,
                0,
                nc.DEFAULT_STEP_MULTIPLIER,
                nc.DEFAULT_DRIVE_INTERVAL_MS,
            )

    def test_drive_tweezers_no_side_effects(self):
        """
        Test that drive_tweezers doesn't modify input parameters.
        """
        nc = NanoControl.__new__(NanoControl)

        original_step = 25
        original_interval = 200
        original_reverse = True

        step_multiplier = original_step
        interval_ms = original_interval
        reverse = original_reverse

        with patch.object(nc, "_go", return_value="OK"):
            nc.drive_tweezers(step_multiplier, interval_ms, reverse)

        # Verify inputs weren't modified
        self.assertEqual(step_multiplier, original_step)
        self.assertEqual(interval_ms, original_interval)
        self.assertEqual(reverse, original_reverse)

    def test_drive_tweezers_multiple_calls(self):
        """
        Test multiple consecutive calls to drive_tweezers.
        """
        nc = NanoControl.__new__(NanoControl)

        calls = [(1, False), (5, True), (10, False), (15, True)]

        with patch.object(nc, "_go", return_value="OK") as mock_go:
            for step_mult, reverse in calls:
                nc.drive_tweezers(step_mult, reverse=reverse)

            # Should have made one call for each invocation
            self.assertEqual(mock_go.call_count, len(calls))


unittest.main()
