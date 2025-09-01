"""
Arduino turntable stepper motor controller via serial communication.

Provides a high-level interface for controlling stepper motor
turntables connected to Arduino devices. Supports precise angle
positioning (0-360°), three speed settings, and bidirectional
rotation with automatic error handling and resource cleanup.

Classes:
    ErrorCode: Arduino firmware error codes (IntEnum)
    TurntableError: Base exception for turntable operations
    TurntableConnectionError: Serial communication failures
    TurntableCommandError: Arduino command errors
    TurntableTimeoutError: Operation timeouts
    Turntable: Main controller class with context management

Protocol:
    Command format: M<speed><signed_angle>E (ASCII)
    Speed codes: '1'=slow, '2'=medium, '3'=fast
    Responses: "\\ACK: ..." or "\\NAK: <code>: <message>"
    Line ending: CRLF (\\r\\n)

Requirements:
    - pyserial
    - Arduino firmware with matching ErrorCode enum

Usage:
    with Turntable('COM3') as tt:
        tt.rotate(90)                    # 90° clockwise, fast
        tt.rotate(45, speed='medium')    # 45° clockwise, medium
        tt.rotate(30, reverse=True)      # 30° counterclockwise

Note:
    Keep ErrorCode synchronized with Arduino firmware tErrorCode enum.
"""

import time
import serial
from types import TracebackType
from typing import Optional, Type, Union
from enum import IntEnum


def tested(func):
    """
    Decorator that marks a function or method as tested by adding a
    'tested' attribute set to True.

    This decorator is used for development/QA purposes to track which
    functions have been verified to work correctly. It can be used
    programmatically to check test coverage or generate reports.

    Args:
        func (callable): The function or method to mark as tested.

    Returns:
        callable: The original function with an added 'tested' attribute
                 set to True.

    Example:
        @tested
        def my_function():
            pass

        # Check if the function is marked as tested:
        if hasattr(my_function, 'tested') and my_function.tested:
            print("Function is tested!")
    """

    func.tested = True
    return func


class ErrorCode(IntEnum):
    """
    Error codes returned by the Arduino turntable firmware.

    These integer codes are sent in NAK responses when the Arduino
    encounters an error during command processing. The values must
    be kept synchronised with the tErrorCode enum in the Arduino
    firmware to ensure proper error handling.

    The error codes are received as single digits in NAK responses
    with format: "\\NAK: <code>: <description>"

    Attributes:
        SUCCESS (0): Command completed successfully
        INVALID_MESSAGE (1): Arduino didn't receive a valid message.
        SYNTAX_ERROR (2): Arduino could not parse the received message.
        TIMEOUT (3): Arduino operation timed out during execution.

    Example:
        # Arduino sends: "\\NAK: 2: Invalid command syntax"
        error_code = ErrorCode(2)  # ErrorCode.SYNTAX_ERROR

    Note:
        If the Arduino firmware's tErrorCode enum is modified, this
        class must be updated to match to prevent protocol errors.
    """

    SUCCESS = 0
    INVALID_MESSAGE = 1
    SYNTAX_ERROR = 2
    TIMEOUT = 3


class TurntableError(Exception):
    """
    Base exception for all turntable-related errors.

    This serves as the parent class for all turntable-specific
    exceptions, allowing users to catch any turntable error with a
    single except clause. All turntable operations should raise
    exceptions derived from this class.

    Example:
        try:
            with Turntable('COM3') as tt:
                tt.rotate(90)
        except TurntableError as e:
            print(f"Turntable operation failed: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Super refers to the parent class, in this case, Exception. By
        calling __init__ on the parent class we are passing the message
        up to Exception.
        """

        super().__init__(message)


class TurntableConnectionError(TurntableError):
    """
    Exception raised when serial communication with the turntable
    device fails.

    Raised when:
        - Serial port cannot be opened
        - Connection to device is lost during operation
        - Device is disconnected or not responding
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class TurntableCommandError(TurntableError):
    """
    Exception raised when the turntable device reports a command
    error.

    Raised when:
        - Arduino returns a NAK response with an unsafe error code.
        - Arduino response is empty or invalid.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class TurntableTimeoutError(TurntableError):
    """
    Exception raised when a turntable operation times out.

    Raised when:
        - Arduino operation exceeds its internal timeout
        - Serial communication timeout occurs
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Turntable:
    """
    Arduino-based turntable stepper motor controller via serial.

    Provides high-level interface for controlling stepper motor
    turntables connected to Arduino devices. Supports precise angle
    positioning (0-360°), three speed settings, and bidirectional
    rotation with automatic error handling and resource cleanup.

    Usage:
        with Turntable('COM3') as tt:
            tt.rotate(90)                    # 90° clockwise, fast
            tt.rotate(45, speed='medium')    # 45° clockwise, medium
            tt.rotate(30, reverse=True)      # 30° counterclockwise

    Raises:
        TurntableConnectionError: Serial communication failures
        TurntableCommandError: Arduino command errors
        TurntableTimeoutError: Operation timeouts
        ValueError: Invalid parameters
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    # Protocol constants (must match Arduino firmware)
    START_BYTE = "M"
    END_BYTE = "E"
    SLOW_SPEED_BYTE = "1"
    MEDIUM_SPEED_BYTE = "2"
    FAST_SPEED_BYTE = "3"

    # Speed mapping
    SPEED_MAPPING = {
        "slow": SLOW_SPEED_BYTE,
        "medium": MEDIUM_SPEED_BYTE,
        "fast": FAST_SPEED_BYTE,
    }

    # Angle limits, max one revolution.
    MIN_ANGLE_DEG = 0.0
    MAX_ANGLE_DEG = 360.0

    # Serial communication settings
    DEFAULT_BAUD_RATE = 9600
    DEFAULT_INIT_SERIAL_DELAY_SEC = 0.2
    DEFAULT_SERIAL_TIMEOUT_SEC = 30  # Larger than Arduino code timeout.

    # Response parsing constants
    ACK_PREFIX = "\\ACK: "
    NAK_PREFIX = "\\NAK: "

    # Validation constants
    ANGLE_PRECISION = 2  # Maximum decimal places for angles

    # -------------------------------------------------------------------------
    # Initialisation functions-------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def __init__(
        self,
        com_port: str,
        baud_rate: int = DEFAULT_BAUD_RATE,
        timeout_sec: Union[int, float] = DEFAULT_SERIAL_TIMEOUT_SEC,
    ) -> None:
        """
        Initialise Turntable interface and establish device connection.

        Opens a serial connection to the specified port and verifies
        that a compatible Arduino turntable device is connected by
        sending a test command.

        Args:
            com_port (str): Serial port identifier (e.g., 'COM3' on
                Windows)
            baud_rate (int, optional): Serial communication baud rate.
                Must match Arduino firmware setting. Defaults to 9600.
            timeout_sec (Union[int, float], optional): Serial timeout in
                seconds for read/write operations. Defaults to
                DEFAULT_SERIAL_TIMEOUT_SEC.

        Raises:
            ValueError: If com_port is empty, baud_rate is not positive,
                or timeout_sec is not positive
            TurntableConnectionError: If anything fails such as the
                serial connection fails or the initialisation handshake.
        """

        # Validate inputs
        if not isinstance(com_port, str) or not com_port.strip():
            raise ValueError(
                "Turntable.__init__: com_port must be a non-empty string."
            )

        if not isinstance(baud_rate, int) or baud_rate <= 0:
            raise ValueError(
                "Turntable.__init__: baud_rate must be a positive integer."
            )

        if not (isinstance(timeout_sec, (int, float)) and timeout_sec > 0):
            raise ValueError(
                "Turntable.__init__: timeout_sec must be a positive number."
            )

        # Store connection parameters
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.timeout_sec = timeout_sec

        # Attempt to connect to Arduino
        try:
            self.device_serial = serial.Serial(
                com_port, baud_rate, timeout=timeout_sec
            )

            # Clear any startup messages
            self.device_serial.flushInput()

            time.sleep(self.DEFAULT_INIT_SERIAL_DELAY_SEC)

            # Clear the Arduino buffer by sending an end byte.
            response = self._send_command(self.END_BYTE)

            # Take the response from the final byte and check the
            # parsed message corresponds to that of an end byte.
            parsed_response = self._parse_response(response)

            if not parsed_response.startswith(f"{ErrorCode.SYNTAX_ERROR}:"):
                raise TurntableConnectionError(
                    "Turntable.__init__: Handshake failed. Expected syntax "
                    f"error, instead got: {parsed_response}"
                )

        # Catch any exception during connection/verification
        except Exception as e:
            self._close_serial()
            raise TurntableConnectionError(
                f"Failed to connect to turntable on {com_port}: {e}"
            ) from e

    @tested
    def __enter__(self) -> "Turntable":
        """
        Context manager entry point for 'with' statement.

        Returns:
            Turntable: This instance for use within the context block
        """

        return self

    @tested
    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[TracebackType],
    ) -> None:
        """
        Context manager exit point.

        Ensures serial port is closed properly regardless of how the
        context is exited (normal completion or exception).

        Args:
            exception_type: Type of exception that caused exit (if any)
            exception_value: Exception instance (if any)
            exception_traceback: Traceback object (if any)
        """

        self._close_serial()

        # Log exception information if an error occurred
        if exception_type is not None:
            print(f"Turntable.__exit__: Exception type: {exception_type}")
            print(f"Turntable.__exit__: Exception value: {exception_value}")
            print(f"Turntable.__exit__: Traceback: {exception_traceback}")

    @tested
    def __str__(self) -> str:
        """
        String representation of the Turntable object.

        Returns:
            str: Human-readable description of the turntable
        """

        return f"Turntable on {self.com_port}"

    # -------------------------------------------------------------------------
    # Private helper functions-------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def _close_serial(self) -> None:
        """
        Safely close the serial port only if it's open.
        """

        try:
            if hasattr(self, "device_serial") and self.device_serial.is_open:
                self.device_serial.close()
                print("Turntable._close_serial: Serial connection closed.")

        except Exception as e:
            print(
                f"Turntable._close_serial: WARNING: Exception when closing "
                f"serial port: {e}"
            )

    @tested
    def _build_command(
        self, speed: str, angle_deg: float, reverse: bool
    ) -> str:
        """
        Build command string according to Arduino protocol.

        Validates inputs and formats them into the protocol command
        string. Negative angles are used for reverse rotation.

        Protocol format: M<speed><angle>E
        Example: M290.00E (medium speed, 90 degrees clockwise)
                M3-45.50E (fast speed, 45.5 degrees counterclockwise)

        Args:
            speed (str): Speed setting ('slow', 'medium', 'fast')
            angle_deg (float): Angle in degrees (0-360)
            reverse (bool): If True, rotate counterclockwise

        Returns:
            str: Formatted command string ready for transmission

        Raises:
            ValueError: If speed is invalid or angle_deg is out of range
        """

        if not isinstance(angle_deg, (int, float)):
            raise ValueError(
                f"Turntable._build_command: 'angle_deg' must be a number, "
                f"got {type(angle_deg)}"
            )

        if not (self.MIN_ANGLE_DEG <= angle_deg <= self.MAX_ANGLE_DEG):
            raise ValueError(
                f"Turntable._build_command: 'angle_deg' must be between "
                f"{self.MIN_ANGLE_DEG} and {self.MAX_ANGLE_DEG} degrees, "
                f"got {angle_deg}"
            )

        if not isinstance(speed, str):
            raise ValueError(
                f"Turntable._build_command: 'speed' must be a string, "
                f"got {type(speed)}"
            )

        if not isinstance(reverse, bool):
            raise ValueError(
                "Turntable._build_command: 'reverse' must be a bool, "
                f"got {type(reverse)}"
            )

        if speed not in self.SPEED_MAPPING:
            valid_speeds = list(self.SPEED_MAPPING.keys())
            raise ValueError(
                f"Turntable._build_command: 'speed' must be one of "
                f"{valid_speeds}, got '{speed}'"
            )

        speed_byte_str = self.SPEED_MAPPING[speed]

        if reverse:
            angle_str = f"{-angle_deg:.{self.ANGLE_PRECISION}f}"
        else:
            angle_str = f"{angle_deg:.{self.ANGLE_PRECISION}f}"

        return f"{self.START_BYTE}{speed_byte_str}{angle_str}{self.END_BYTE}"

    @tested
    def _send_command(self, command: str) -> str:
        """
        Send command to Arduino and return the response.

        Transmits the command via serial, waits for response, and
        handles communication errors and timeouts.

        Args:
            command (str): Command string formatted per Arduino protocol
                (e.g., "M290.00E")

        Returns:
            str: Raw response message from Arduino device

        Raises:
            TurntableTimeoutError: No response received within timeout
            TurntableConnectionError: Serial communication failure
            ValueError: Command contains non-ASCII characters
        """

        try:
            # Encode command
            command_bytes = command.encode("ascii")

            # Clear input buffer and send command
            self.device_serial.flushInput()
            self.device_serial.write(command_bytes)
            self.device_serial.flush()

            # Wait for a response from the device up to and including
            # the carriage return. Decode the received bytes into a
            # string using ASCII
            response = (
                self.device_serial.read_until(b"\r\n").decode("ascii").strip()
            )

            # Timeout occurred
            if not response:
                raise TurntableTimeoutError(
                    f"Turntable._send_command: No response received within "
                    f"{self.timeout_sec} seconds for command: {command}"
                )

            return response

        except UnicodeEncodeError as e:
            raise ValueError(
                f"Turntable._send_command: Command contains non-ASCII "
                f"character: {e}"
            ) from e

        except serial.SerialException as e:
            raise TurntableConnectionError(
                f"Turntable._send_command: Serial communication error: {e}"
            ) from e

    @tested
    def _parse_response(self, response: str) -> str:
        """
        Parse Arduino response and handle ACK/NAK status.

        Processes response strings from Arduino, extracts status and
        error codes, and raises appropriate exceptions for errors.

        Arduino response format:
            Success: \\ACK: Finished executing command: <command>
            Error:   \\NAK: <error_code>: <error_description>

        Args:
            response (str): Raw response from Arduino

        Returns:
            str: Success message for ACK responses and error message
                for safe fails from NAK responses

        Raises:
            TurntableCommandError: For NAK responses or parse errors
            TurntableTimeoutError: For timeout error codes
        """

        if not isinstance(response, str) or not response:
            raise TurntableCommandError(
                "Turntable._parse_response: Invalid or empty response."
            )

        # Parse ACK response; Extract and return message after "\\ACK: "
        if response.startswith(self.ACK_PREFIX):
            return response[len(self.ACK_PREFIX) :].strip()

        # Handle NAK response
        elif response.startswith(self.NAK_PREFIX):
            # Extract error code (first character after "\\NAK: ")
            error_code = int(response[len(self.NAK_PREFIX) :].strip()[0])

            # Extract full error message for user
            error_message = response[len(self.NAK_PREFIX) :].strip()

            # Safe fails, no need to raise - just send another command.
            if (error_code == ErrorCode.INVALID_MESSAGE) or (
                error_code == ErrorCode.SYNTAX_ERROR
            ):
                print(f"Turntable._parse_response: {error_message}")
                return error_message

            elif error_code == ErrorCode.TIMEOUT:
                raise TurntableTimeoutError(
                    "Turntable._parse_response: Arduino timed out: "
                    f"{error_message}"
                )
            else:
                raise TurntableCommandError(
                    "Turntable._parse_response: Unknown Arduino error: "
                    f"{error_message}"
                )

        # Received neither ACK nor NAK
        else:
            raise TurntableCommandError(f"Unknown response format: {response}")

    # -------------------------------------------------------------------------
    # Public Interface---------------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def rotate(
        self,
        angle_deg: float,
        speed: str = "fast",
        reverse: bool = False,
    ) -> str:
        """
        Rotate the turntable by the specified angle at the given speed.

        Sends rotation command to Arduino and waits for completion.
        Supports three speed settings with acceleration for smooth
        motion.

        Args:
            angle_deg (float): Angle to rotate in degrees. Must be
                positive (0 to 360 degrees).
            speed (str, optional): Rotation speed setting. Must be one
                of: 'slow', 'medium', 'fast'. Defaults to 'fast'. There
                is an acceleration feature to provide smooth motion so
                it may not reach max speed if the angle is small.
            reverse (bool, optional): If True, rotate counterclockwise.
                If False, rotate clockwise. Defaults to False.

        Returns:
            str: Success message for ACK responses and error message
                for safe fails from NAK responses

        Raises:
            ValueError: If angle_deg is outside valid range (0 to 360)
                or speed and reverse are not valid options.
            TurntableCommandError: If the Arduino reports an error during
                execution (syntax error, invalid command, etc.).
            TurntableTimeoutError: If the rotation operation times out.
            TurntableConnectionError: If serial communication fails.

        Example:
            # Rotate 90 degrees clockwise at medium speed
            tt.rotate(90)

            # Rotate 45.5 degrees counterclockwise at fast speed
            tt.rotate(45.5, speed='fast', reverse=True)
        """

        try:
            command = self._build_command(speed, angle_deg, reverse)

        except ValueError as e:
            # Strip internal method name from error message so they
            # appear clean to the user.
            msg = str(e)

            if msg.startswith("Turntable._build_command: "):
                clean_msg = msg[len("Turntable._build_command: ") :]
                raise ValueError(f"Turntable.rotate: {clean_msg}") from e
            else:
                raise  # Re-raise unchanged if unexpected format

        response = self._send_command(command)
        return self._parse_response(response)
