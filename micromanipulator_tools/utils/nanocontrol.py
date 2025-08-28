"""
NanoControl Interface Module
----------------------------

Provides an interface to the Kleindiek mm3a micromanipulator (no
encoders) via serial communication. Supports safe usage through Python's
context manager protocol (`with`), ensuring automatic cleanup of the
serial connection.

Classes:
    NanoControl: Main control class for interacting with the manipulator
    NanoControlError: Base exception for NanoControl-related errors
    NanoControlCommandError: Exception for device command errors
    NanoControlConnectionError: Exception for serial communication
        issues

Features:
    Serial communication with device protocol handling
    Speed profile configuration and management, not to be confused with
        NanoControl user/configuration profiles, read page 13 in the
        operation manual for information on the latter.
    Individual joint movement control
    Safe resource management via context managers
    Comprehensive error handling and validation

Usage Example:

    with NanoControl('COM3') as nc:
        # Get device information
        version = nc.get_version()

        # Configure speed profile
        speeds = {
            "base_joint": "c32",
            "elbow_joint": "c32",
            "prismatic_joint": "c32",
            "tweezer_joint": "c32"
        }
        nc.set_speed_profile(1, speeds)

        # Move joints
        nc.drive_base_joint(step_multiplier=5, reverse=True)
        nc.stop()

Requirements:
    - Python 3.8+
    - pyserial library
    - Kleindiek mm3a micromanipulator hardware

Important Configuration Notes
-----------------------------
Speed Profiles vs Device Configuration Profiles:

This module manages speed profiles (temporary settings), which are
different from the NanoControl's built-in user/configuration profiles.
See page 13 of the operation manual for details on device profiles.

Precision Configuration:

This implementation was developed for relatively large-scale movements
and has not been optimized for nanometer precision work. For
high-precision applications:

1. Configure proper minimum amplitude settings (see operation manual)
2. Set up a dedicated NanoControl configuration profile for your project
3. Use the device's flash memory to store persistent settings
4. The device supports 6 configuration profiles, each with 6 speed
   profiles amongst other settings

Current Setup:

• Uses NanoControl configuration profile #1
• Configured for large amplitude movements
• Settings are not optimized for microscope-assisted precision work

Recommendation:

For micro-scale work, create a separate device configuration profile to
avoid conflicts and enable nanometer precision capabilities with
microscope integration.

-----------------------------
Author: Nick Agathangelou <nick.agathangelou@ext.esa.int>
Version: 1.0.0
"""

import re
import time
import serial
from types import TracebackType
from typing import Optional, Type, Union, List


def tested(func):
    """
    Decorator that marks a function or method as tested by adding a
    'tested' attribute set to True.

    This can be used to programmatically check whether a function has
    been confirmed to work as expected.

    Args:
        func (callable): The function or method to mark as tested.

    Returns:
        callable: The original function with a 'tested' attribute.

    Usage:
        @tested
        def myFunction():
            pass

        # Later, check if the method is tested:
        if hasattr(obj.myFunction, 'tested'):
            print("Method is tested!")
    """

    func.tested = True
    return func


class NanoControlError(Exception):
    """
    Base exception for all NanoControl-related errors.

    This is the parent class for all NanoControl-specific exceptions.
    Catch this to handle any NanoControl error generically.

    Example:
        try:
            with NanoControl('COM3') as nc:
                nc.get_version()
        except NanoControlError as e:
            print(f"NanoControl error: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Super refers to the parent class, in this case, Exception. By
        calling __init__ on the parent class we are passing the message
        up to Exception.
        """

        super().__init__(message)


class NanoControlConnectionError(NanoControlError):
    """
    Exception for NanoControl connection issues. Derived from
    NanoControlError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class NanoControlCommandError(NanoControlError):
    """
    Exception for NanoControl command processing issues. Derived from
    NanoControlError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class NanoControl:
    """
    Controls a Kleindiek mm3a micromanipulator (no encoders) via serial
    interface.

    This class provides high-level methods to send movement and control
    commands to the manipulator hardware.

    Usage:
        with NanoControl('COM19') as nc:
            nc.get_version()

            # Configure speed profile
            speeds = {
                "base_joint": "c32",
                "elbow_joint": "c32",
                "prismatic_joint": "c32",
                "tweezer_joint": "c32"
            }
            nc.set_speed_profile(1, speeds)

    Note:
        Always use in a `with` block to ensure the serial port is
        closed properly.
        Do not call private methods (prefixed with `_`) directly from
        outside the class.
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    # NanoControl speed constants
    MIN_SPEED_ARG = 1
    MAX_SPEED_ARG = 64
    MIN_SPEED_PROFILE_NUM = 1
    MAX_SPEED_PROFILE_NUM = 6
    MIN_STEP_MULTIPLIER = -100
    MAX_STEP_MULTIPLIER = 100

    DEFAULT_DRIVE_INTERVAL_MS = 10
    DEFAULT_STEP_MULTIPLIER = 1

    DEFAULT_BAUD_RATE = 115200
    DEFAULT_TIMEOUT_SEC = 10

    # Initialisation beep settings
    INIT_BEEP_FREQ = 1500
    INIT_BEEP_DURATION_MS = 150
    INIT_BEEP_INTERVAL_SEC = 0.05
    INIT_NUM_BEEPS = 3

    # Default motor drive frequencies for joints A, B, C, D (Hz)
    DEFAULT_MOTOR_FREQUENCIES = [1000, 1000, 1000, 1000]

    # Valid frequency range for each joint and the NanoControl
    # command module. (Hz)
    MIN_FREQUENCY = 300
    MAX_FREQUENCY = 20000

    # Joint identification
    JOINT_NAMES = [
        "base_joint",
        "elbow_joint",
        "prismatic_joint",
        "tweezer_joint",
    ]

    EXPECTED_SPEED_RESPONSE_TOKENS = 5
    SPEED_PREFIX_LENGTH = 1

    # -------------------------------------------------------------------------
    # Initialisation functions-------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def __init__(
        self,
        com_port: str,
        baud_rate: int = DEFAULT_BAUD_RATE,
        timeout_sec: Union[int, float] = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        """
        Initialize NanoControl interface and establish device connection.

        Opens a serial connection to the specified port and verifies
        that a compatible NanoControl device is connected by:

        1. Setting default motor frequencies
        2. Enabling fine-with-coarse mode
        3. Sending verification beeps to confirm device response

        Args:
            com_port (str): Serial port identifier (e.g., 'COM3' on
                Windows)
            baud_rate (int, optional): Serial communication baud rate.
                Defaults to 115200.
            timeout_sec (Union[int, float], optional): Serial timeout in
                seconds for read/write operations. Defaults to 10.

        Raises:
            ValueError: If com_port is empty, baud_rate is not positive,
                or timeout_sec is not positive
            NanoControlError: If serial connection fails or device
                verification fails (device not responding or
                incompatible)
        """
        # Raise errors if the inputs do not have the correct format or
        # do not make sense.
        if not isinstance(com_port, str) or not com_port.strip():
            raise ValueError(
                "NanoControl.__init__: com_port must be a non-empty string."
            )

        if not isinstance(baud_rate, int) or baud_rate <= 0:
            raise ValueError(
                "NanoControl.__init__: baud_rate must be a positive integer."
            )

        if not (isinstance(timeout_sec, (int, float)) and timeout_sec > 0):
            raise ValueError(
                "NanoControl.__init__: timeout_sec must be a positive number."
            )

        # Store the inputs as instance attributes so other methods in
        # the class can access them.
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.timeout_sec = timeout_sec

        # Attempt to connect to NanoControl
        try:
            self.device_serial = serial.Serial(
                com_port, baud_rate, timeout=timeout_sec
            )
            self.set_motor_drive_frequencies(self.DEFAULT_MOTOR_FREQUENCIES)
            self.set_fine_with_coarse_all(True)

            # Beep multiple times to verify the device is a NanoControl.
            # This will throw an error through _send_command if the
            # device does not behave like a NanoControl.
            for i in range(self.INIT_NUM_BEEPS):
                self.beep(self.INIT_BEEP_FREQ, self.INIT_BEEP_DURATION_MS)
                time.sleep(self.INIT_BEEP_INTERVAL_SEC)

        except serial.SerialException as e:
            self._close_serial()
            raise NanoControlError(
                f"NanoControl.__init__: Serial connection "
                f"failed on {com_port}."
            ) from e

        except (
            NanoControlConnectionError,
            NanoControlCommandError,
            ValueError,
        ) as e:
            self._close_serial()
            raise NanoControlError(
                f"NanoControl.__init__: Device verification "
                f"failed on {com_port}."
            ) from e

    @tested
    def __enter__(self) -> "NanoControl":
        """
        When the class is called using a 'with' block, it assigns
        whatever __enter__ returns to the variable after the 'as' in the
        'with NanoControl as myObject' instantiation.
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
        Context manager exit point. Ensures serial port is closed
        properly regardless of how the context is exited.

        Args:
            exception_type: Type of exception that caused exit (if any)
            exception_value: Exception instance (if any)
            exception_traceback: Traceback object (if any)
        """

        self._close_serial()

        # Log exception information if an error occurred
        if exception_type is not None:
            print(f"NanoControl.__exit__: Exception type: {exception_type}")
            print(f"NanoControl.__exit__: Exception value: {exception_value}")
            print(f"NanoControl.__exit__: Traceback: {exception_traceback}")

    @tested
    def __str__(self) -> str:
        """
        String representation of the NanoControl object.

        Returns:
            str: Human-readable description of the turntable
        """

        return f"NanoControl.__str__: NanoControl Port: {self.com_port}"

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
                print("NanoControl._close_serial: Serial connection closed.")

        # Do not raise an exception as this is a safe fail.
        except Exception as e:
            print(
                f"NanoControl._close_serial: WARNING: Exception when closing "
                f"serial port: {e}"
            )

    @tested
    def _send_command(self, command_str: str) -> str:
        """
        Send a command string to the NanoControl device via serial,
        handle its response, and raise exceptions on errors or timeouts.

        Args:
            command_str (str): Command formatted as per NanoControl
                protocol.

        Returns:
            str: The message content from the NanoControl device if the
                response status is 'o' (OK) or 'i' (info).

        Raises:
            NanoControlError: If the device returns an error
                status, an unknown response, no response (timeout), or
                if there is a serial communication error.
        """

        # Use the try and except to protect from unexpected serial
        # issues.
        try:
            # Append a carriage return to the end of the command and
            # encode using ascii to send the characters as bytes.
            command_bytes = (command_str + "\r").encode("ascii")

            # Clear the input buffer.
            self.device_serial.flushInput()

            # Write the command to the serial.
            self.device_serial.write(command_bytes)

            # Flush the buffer and wait for the answer.
            self.device_serial.flush()

            # Wait for a response from the device up to and including
            # the next carriage return. Decode the recieved bytes into a
            # string using ASCII
            response = (
                self.device_serial.read_until(b"\r").decode("ascii").strip()
            )

            # The serial port was opened with a timeout. If no byte has
            # been received before timeout, raise a timeout error.
            if not response:
                raise NanoControlCommandError(
                    "NanoControl._send_command: No response from device (timeout)"
                )

            # Separate the response message at the tabs.
            status_char, _, message = response.partition("\t")

            # Message is OK.
            if status_char == "o":
                return message

            # Message has an error.
            elif status_char == "e":
                raise NanoControlCommandError(
                    f"NanoControl._send_command: Device error: {message}"
                )

            # Message has information.
            elif status_char == "i":
                print(
                    f"NanoControl._send_command: Info from device: {message}"
                )
                return message

            # The initial response could not be decoded.
            else:
                raise NanoControlCommandError(
                    f"NanoControl._send_command: Unknown response status: {response}"
                )

        # Catch potential ascii encoding errors
        except UnicodeEncodeError as e:
            # Re-raise as a more user-friendly ValueError. The original
            # error 'e' has a very descriptive message.
            raise ValueError(
                f"NanoControl._send_command: Command string contains a "
                f"non-ASCII character. {e}"
            ) from e

        except serial.SerialException as e:
            # Re-raise the custom exception, chaining the original for
            # a full traceback.
            raise NanoControlConnectionError(
                f"NanoControl._send_command: Serial error: {e}"
            ) from e

    @tested
    def _query_speed_profile(self) -> str:
        """
        Query the current active speed profile from the device.

        Returns:
            str: Raw response from the device
                (e.g., "1 c64 c64 c64 c64").

        Raises:
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.
        """

        return self._send_command("speed ?")

    @tested
    def _validate_speed_value(
        self, speed_str: str, allow_device_format: bool = False
    ) -> None:
        """
        Validates a speed configuration string.

        Supports two formats:
            User input format (3 chars): 'c32', 'f08'
                i.e. letter + 2 digits
            Device response format (4 chars): 'c053', 'f001'
                i.e. letter + 3 digits (zero-padded)

        Args:
            speed_str (str): The speed string to validate.
            allow_device_format (bool): If True, accepts 4-character
                device format. If False, only accepts 3-character user
                format. Defaults to False.

        Raises:
            ValueError: If the string format is invalid or the speed
                value is out of range.
        """

        if not isinstance(speed_str, str):
            raise ValueError(
                f"NanoControl._validate_speed_value: Speed value must be "
                f"a string, got {type(speed_str)}"
            )

        speed_lower = speed_str.lower()

        # Check format based on allowed formats
        if allow_device_format:
            # Accept both 3-char (c32) and 4-char (c053) formats
            if not re.match(r"^[cf]\d{2,3}$", speed_lower):
                raise ValueError(
                    f"NanoControl._validate_speed_value: Speed value "
                    f"'{speed_str}' is not in valid format. "
                    "Expected 3-char (e.g., 'c32') or 4-char (e.g., 'c053') "
                    "format."
                )
        else:
            # Only accept 3-char format (c32)
            if not re.match(r"^[cf]\d{2}$", speed_lower):
                raise ValueError(
                    f"NanoControl._validate_speed_value: Speed value "
                    f"'{speed_str}' is not in valid 3-character "
                    "format (e.g., 'c32', 'f08')."
                )

        # Extract and validate the numeric part (works for both 2 and 3
        # digit formats)
        speed_value = int(speed_str[self.SPEED_PREFIX_LENGTH :])
        if not (self.MIN_SPEED_ARG <= speed_value <= self.MAX_SPEED_ARG):
            raise ValueError(
                f"NanoControl._validate_speed_value: Speed value "
                f"{speed_value} in '{speed_str}' must be "
                f"between {self.MIN_SPEED_ARG} and {self.MAX_SPEED_ARG}."
            )

    @tested
    def _validate_profile_number(self, profile_num: int) -> None:
        """
        Validates that a profile number is an integer in the valid
        range from MIN_SPEED_PROFILE_NUM to MAX_SPEED_PROFILE_NUM

        Args:
            profile_num (int): Profile number to validate.

        Raises:
            ValueError: If profile_num is not an integer or outside
                range 1-6.
        """

        if type(profile_num) is not int or not (
            self.MIN_SPEED_PROFILE_NUM
            <= profile_num
            <= self.MAX_SPEED_PROFILE_NUM
        ):
            raise ValueError(
                "NanoControl._validate_profile_number: speed profile number "
                f"must be an integer from {self.MIN_SPEED_PROFILE_NUM} to "
                f"{self.MAX_SPEED_PROFILE_NUM}."
            )

    @tested
    def _parse_speed_response(self, response: str) -> tuple[int, list[str]]:
        """
        Parse a speed profile response from the device.

        Parses the device response into a profile number and list of
        speed values for each joint. Validates that the response format
        is correct and all speed values are valid.

        Args:
            response (str): Raw device response containing profile
                number and 4 speed values
                (e.g., "1 c064 c032 f008 c016").

        Returns:
            tuple[int, list[str]]:
                (profile_number, [base_speed, elbow_speed,
                prismatic_speed, tweezer_speed])

        Raises:
            NanoControlError: Raised in the following specific cases:
                Response format error: If the response does not contain
                    exactly 5 space-separated tokens (1 profile number
                    + 4 speeds). Example: "1 c32 f16" (only 3 tokens
                    instead of 5).
                Profile number parsing error: If the first token cannot
                    be converted to an integer or is outside the valid
                    range (1-6). Examples: "abc c32 f16 c08 c64" (not
                    an integer), "0 c32 f16 c08 c64" (out of range).
                Speed value format error: If any of the 4 speed tokens
                    do not match the expected format (letter + 2-3
                    digits, where letter is 'c' or 'f' and digits
                    represent 1-64). Examples: "1 x99 f16 c08 c64"
                    (invalid prefix 'x'), "1 c00 f16 c08 c64" (speed
                    value 0 out of range), "1 c999 f16 c08 c64" (wrong
                    length - 4 characters). The error will specify which
                    joint (base, elbow, prismatic, or tweezer) has the
                    invalid speed value.

        Note: All NanoControlError instances are chained to the
        original ValueError that caused the parsing/validation
        failure, preserving the full error context for debugging.
        """

        # Split response into tokens
        parts = response.split()

        # Validate we have exactly 5 parts: profile + 4 speeds
        if len(parts) != self.EXPECTED_SPEED_RESPONSE_TOKENS:
            raise NanoControlError(
                f"NanoControl._parse_speed_response: Expected "
                f"{self.EXPECTED_SPEED_RESPONSE_TOKENS} tokens (profile + "
                f"{len(self.JOINT_NAMES)} speeds), got {len(parts)}. "
                f"Response: {response!r}"
            )

        # Parse and validate profile number
        try:
            profile_number = int(parts[0])
            self._validate_profile_number(profile_number)

        except ValueError as error:
            raise NanoControlError(
                f"NanoControl._parse_speed_response: Profile number "
                f"'{parts[0]}' is not an int in the valid range. "
                f"Response: {response!r}"
            ) from error

        # Parse and validate speed values
        speeds = parts[1:]

        for index, speed_token in enumerate(speeds):
            try:
                self._validate_speed_value(speed_token, True)

            except ValueError as error:
                if index < len(self.JOINT_NAMES):
                    joint_name = self.JOINT_NAMES[index]
                else:
                    joint_name = f"joint_{index + 1}"

                raise NanoControlError(
                    f"NanoControl._parse_speed_response: Invalid {joint_name} "
                    f"speed '{speed_token}': {error}. Response: {response!r}"
                ) from error

        return profile_number, speeds

    @tested
    def _validate_go_steps(self, step_multiplier: int) -> None:
        """
        A private method to check whether the step multiplier requested to
        drive the joints is valid. This means it is an integer in the
        range (MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER)

        Args:
            step_multiplier (int): Step multiplier for speed profile steps.

        Raises:
            ValueError: If step_multiplier is not an integer in the range
                MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER.
        """

        if type(step_multiplier) is not int:  # Changed from isinstance
            raise ValueError(
                "NanoControl._validate_go_steps: step_multiplier must "
                "be an int."
            )

        if not (
            self.MIN_STEP_MULTIPLIER
            <= step_multiplier
            <= self.MAX_STEP_MULTIPLIER
        ):
            raise ValueError(
                "NanoControl._validate_go_steps: step_multiplier must be "
                f"between {self.MIN_STEP_MULTIPLIER} and {self.MAX_STEP_MULTIPLIER}."
            )

    @tested
    def _go(
        self,
        step_multiplier_a: int,
        step_multiplier_b: int,
        step_multiplier_c: int,
        step_multiplier_d: int,
        interval_ms: int,
    ) -> str:
        """
        Private method to send the go command to the device.

        Total steps executed per interval = (speed profile steps) x
        (step_multiplier). For example, if speed profile is 'c32' and
        step_multiplier=5, then 160 steps are executed per interval.

        Args:
            step_multiplier_a, step_multiplier_b, step_multiplier_c,
            step_multiplier_d (int): Step multipliers for joints A-D
                (MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER)
            interval_ms (int): Time interval between step executions in
                milliseconds

        Returns:
            str: Device response

        Raises:
            ValueError: If any parameter is invalid
            NanoControlConnectionError: If serial communication fails
            NanoControlCommandError: If device returns an error
        """

        # Validate each step multiplier using the dedicated validator
        for i, step_multiplier in enumerate(
            [
                step_multiplier_a,
                step_multiplier_b,
                step_multiplier_c,
                step_multiplier_d,
            ]
        ):
            try:
                self._validate_go_steps(step_multiplier)
            except ValueError as e:
                # Extract the joint name for better error messages
                joint_name = (
                    self.JOINT_NAMES[i]
                    if i < len(self.JOINT_NAMES)
                    else f"joint_{i + 1}"
                )
                # Re-raise with joint-specific context
                error_msg = (
                    str(e).split(": ", 1)[1] if ": " in str(e) else str(e)
                )
                raise ValueError(
                    f"NanoControl._go: {joint_name} {error_msg}"
                ) from e

        # Validate interval
        if type(interval_ms) is not int or interval_ms <= 0:
            raise ValueError(
                f"NanoControl._go: interval_ms must be a positive integer, "
                f"got {interval_ms}"
            )

        command = (
            f"go {step_multiplier_a} {step_multiplier_b} "
            f"{step_multiplier_c} {step_multiplier_d} {interval_ms}"
        )
        return self._send_command(command)

    @tested
    def _process_step_multiplier_and_direction(
        self, step_multiplier: int, reverse: bool
    ) -> int:
        """
        Process step multiplier and apply direction, with warning for
        negative values.

        Args:
            step_multiplier (int): The step multiplier value
            reverse (bool): If True, make result negative

        Returns:
            int: Processed step multiplier with direction applied
        """
        abs_multiplier = abs(step_multiplier)
        if step_multiplier < 0:
            print(
                f"NanoControl: WARNING: step_multiplier "
                f"should be positive. Using abs({step_multiplier})="
                f"{abs_multiplier}. Use reverse=True for reverse direction."
            )

        # Apply direction
        final_multiplier = -abs_multiplier if reverse else abs_multiplier
        return final_multiplier

    # -------------------------------------------------------------------------
    # Public Interface---------------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def set_speed_profile(
        self, profile_num: int, speed_profile: dict[str, str]
    ) -> str:
        """
        Configures one of the six speed profiles on the NanoControl
        device.

        Args:
            profile_num (int): The profile number to configure (1-6).
            speed_profile (dict[str, str]): Dictionary with joint
                speeds. Must contain: "base_joint", "elbow_joint",
                "prismatic_joint", "tweezer_joint"

        Returns:
            str: Response message from the device.

        Raises:
            ValueError: If inputs are invalid.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If device returns an error.
            NanoControlError: If unexpected communication error occurs.

        Note:
            The configured speed profile persists until overwritten by
            another speed profile or the device is powered off.
        """

        # Validate profile number
        self._validate_profile_number(profile_num)

        # Validate speed_profile is a dictionary
        if not isinstance(speed_profile, dict):
            raise ValueError(
                "NanoControl.set_speed_profile: speed_profile must "
                "be a dictionary"
            )

        required_joints = self.JOINT_NAMES

        for joint in required_joints:
            if joint not in speed_profile:
                raise ValueError(
                    f"NanoControl.set_speed_profile: Missing "
                    f"required key: {joint}"
                )

        # Validate speed values
        for joint, speed in speed_profile.items():
            if joint in required_joints:
                self._validate_speed_value(speed)

        # Build command: speed <profile> <A> <B> <C> <D>
        command_str = (
            f"speed {profile_num} "
            f"{speed_profile['base_joint']} "
            f"{speed_profile['elbow_joint']} "
            f"{speed_profile['prismatic_joint']} "
            f"{speed_profile['tweezer_joint']}"
        )

        return self._send_command(command_str)

    @tested
    def get_speed_profile(
        self, profile_num: int | None = None
    ) -> dict[str, str]:
        """
        Queries the specified speed profile for the speed settings of
        each Joint. If no profile number is specified, the current
        active profile is queried.

        Note:
            This method may temporarily switch the device to a different
            speed profile for querying, but will restore the original
            active profile before returning.

        Args:
            profile_num (int, optional): The profile number (1-6)
                to query. If None, the currently active profile is used.

        Returns:
            dict[str, str]: A dictionary mapping joint names to their
                respective speed settings.
            Example:
                {
                    "base_joint": "c64",
                    "elbow_joint": "c64",
                    "prismatic_joint": "c64",
                    "tweezer_joint": "c64"
                }
        Raises:
            ValueError: If profile_num is not between 1 and 6.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error.
            NanoControlError: If profile switching verification fails
                or other issues occur.
        """

        original_profile_num: int | None = None

        if profile_num is not None:
            self._validate_profile_number(profile_num)

        try:
            # Get the current active profile and its speeds
            response_0 = self._query_speed_profile()
            original_profile_num, speeds_0 = self._parse_speed_response(
                response_0
            )

            # Decide which profile to use
            if profile_num is None:
                target_profile = original_profile_num
            else:
                target_profile = profile_num

            # Switch profiles and get speeds from the target profile if
            # desired profile is not active.
            if target_profile != original_profile_num:
                _, speeds = self.change_speed_profile_to(target_profile)
            else:
                speeds = speeds_0

            return {
                joint_name: speeds[i]
                for i, joint_name in enumerate(self.JOINT_NAMES)
            }

        except NanoControlError:
            raise

        except Exception as e:
            raise NanoControlError(
                f"NanoControl.get_speed_profile: Unexpected failure: {e}"
            ) from e

        # Defensive restore happens no matter what.
        finally:
            if original_profile_num is not None:
                try:
                    current_response = self._query_speed_profile()
                    current_profile_num, _ = self._parse_speed_response(
                        current_response
                    )
                    if current_profile_num != original_profile_num:
                        self.change_speed_profile_to(original_profile_num)
                except Exception as restore_err:
                    print(
                        f"NanoControl.get_speed_profile: WARNING: "
                        f"failed to restore profile {original_profile_num}: "
                        f"{restore_err}"
                    )

    @tested
    def change_speed_profile_to(
        self, profile_num: int
    ) -> tuple[int, list[str]]:
        """
        Change the device to the use the specified speed profile and
        verify the switch.

        Args:
            profile_num (int): The profile number to switch to (1-6).

        Returns:
            tuple[int, list[str]]: Confirmed profile number and list of
                4 speed strings.

        Raises:
            ValueError: If profile_num is not an integer from 1 to 6.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.
            NanoControlError: If the switch command succeeds but the
                device doesn't report the expected profile, or other
                unexpected failures occur.

        Note:
            The active speed profile persists until changed to another
            speed profile or the device is powered off.
        """

        self._validate_profile_number(profile_num)

        try:
            # Send the switch command
            self._send_command(f"speed {profile_num}")

            # Verify the switch worked
            response = self._query_speed_profile()
            reported_profile_num, speeds = self._parse_speed_response(response)

            # The switch hasn't worked for some reason.
            if reported_profile_num != profile_num:
                raise NanoControlError(
                    "NanoControl.change_speed_profile_to: Device did not "
                    "switch to the requested profile. "
                    f"Requested {profile_num}, device reports "
                    f"{reported_profile_num}. Response: {response!r}"
                )

            return reported_profile_num, speeds

        except NanoControlError:
            raise

        except Exception as e:
            raise NanoControlError(
                "NanoControl.change_speed_profile_to: Failed to change "
                f"speed profile to {profile_num}: {e}"
            ) from e

    @tested
    def drive_base_joint(
        self,
        step_multiplier: int = DEFAULT_STEP_MULTIPLIER,
        interval_ms: int = DEFAULT_DRIVE_INTERVAL_MS,
        reverse: bool = False,
    ) -> str:
        """
        Continuously drive the base joint (channel A) at the specified
        rate. Calling this, or any other drive function will terminate
        any previously running drive function.

        The total steps executed per interval equals the current speed
        profile setting multiplied by the step_multiplier parameter.
        For example, if the speed profile is set to 'c32' and
        step_multiplier=10, then 320 coarse steps will be executed
        every interval_ms milliseconds.

        Args:
            step_multiplier (int, optional): Multiplier for speed
                profile steps. Should be positive (1 to
                MAX_STEP_MULTIPLIER). If negative, absolute value will
                be used with a warning. Defaults to
                DEFAULT_STEP_MULTIPLIER.
                Total steps per interval = (speed profile steps) x
                step_multiplier
            interval_ms (int, optional): Time interval between movement
                executions in milliseconds. Smaller values = faster
                movement. Defaults to DEFAULT_DRIVE_INTERVAL_MS.
            reverse (bool, optional): If True, drive in reverse
                direction (clockwise). If False, drive anticlockwise.
                Defaults to False.

        Returns:
            str: Response message from the device confirming the
                command.

        Raises:
            ValueError: If step_multiplier is not an integer in the
                range MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER or if
                interval_ms is not a positive integer.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Example:
            # With speed profile base_joint = "c32":
            nc.drive_base_joint()              # 32x1=32 steps forward
            nc.drive_base_joint(5)             # 32x5=160 steps forward
            nc.drive_base_joint(reverse=True)  # 32x1=32 steps reverse
            nc.drive_base_joint(2, 100, True)  # 32x2=64 steps reverse
        """

        final_multiplier = self._process_step_multiplier_and_direction(
            step_multiplier, reverse
        )
        return self._go(final_multiplier, 0, 0, 0, interval_ms)

    @tested
    def drive_elbow_joint(
        self,
        step_multiplier: int = DEFAULT_STEP_MULTIPLIER,
        interval_ms: int = DEFAULT_DRIVE_INTERVAL_MS,
        reverse: bool = False,
    ) -> str:
        """
        Continuously drive the elbow joint (channel B).

        Args:
            step_multiplier (int, optional): Multiplier for speed
                profile steps. Defaults to DEFAULT_STEP_MULTIPLIER.
            interval_ms (int, optional): Time interval in milliseconds.
                Defaults to DEFAULT_DRIVE_INTERVAL_MS.
            reverse (bool, optional): If True, drive upwards.
                Defaults to False.

        Returns:
            str: Device response confirming the command.

        Raises:
            ValueError: If step_multiplier is not an integer in the
                range MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER or if
                interval_ms is not a positive integer.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Note:
            See drive_base_joint() for detailed documentation.
        """

        final_multiplier = self._process_step_multiplier_and_direction(
            step_multiplier, reverse
        )
        return self._go(0, final_multiplier, 0, 0, interval_ms)

    @tested
    def drive_prismatic_joint(
        self,
        step_multiplier: int = DEFAULT_STEP_MULTIPLIER,
        interval_ms: int = DEFAULT_DRIVE_INTERVAL_MS,
        reverse: bool = False,
    ) -> str:
        """
        Continuously drive the prismatic joint (channel C).

        Args:
            step_multiplier (int, optional): Multiplier for speed
                profile steps. Defaults to DEFAULT_STEP_MULTIPLIER.
            interval_ms (int, optional): Time interval in milliseconds.
                Defaults to DEFAULT_DRIVE_INTERVAL_MS.
            reverse (bool, optional): If True, drive in reverse.
                Defaults to False.

        Returns:
            str: Device response confirming the command.

        Raises:
            ValueError: If step_multiplier is not an integer in the
                range MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER or if
                interval_ms is not a positive integer.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Note:
            See drive_base_joint() for detailed documentation.
        """
        final_multiplier = self._process_step_multiplier_and_direction(
            step_multiplier, reverse
        )
        return self._go(0, 0, final_multiplier, 0, interval_ms)

    @tested
    def drive_tweezer_joint(
        self,
        step_multiplier: int = DEFAULT_STEP_MULTIPLIER,
        interval_ms: int = DEFAULT_DRIVE_INTERVAL_MS,
        reverse: bool = False,
    ) -> str:
        """
        Continuously drive the tweezer joint (channel D).

        Args:
            step_multiplier (int, optional): Multiplier for speed
                profile steps. Defaults to DEFAULT_STEP_MULTIPLIER.
            interval_ms (int, optional): Time interval in milliseconds.
                Defaults to DEFAULT_DRIVE_INTERVAL_MS.
            reverse (bool, optional): If True, drive in reverse.
                Defaults to False.

        Returns:
            str: Device response confirming the command.

        Raises:
            ValueError: If step_multiplier is not an integer in the
                range MIN_STEP_MULTIPLIER to MAX_STEP_MULTIPLIER or if
                interval_ms is not a positive integer.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Note:
            See drive_base_joint() for detailed documentation.
        """

        final_multiplier = self._process_step_multiplier_and_direction(
            step_multiplier, reverse
        )
        return self._go(0, 0, 0, final_multiplier, interval_ms)

    @tested
    def stop(self) -> str:
        """
        Stops any command that is currently being executed on the
        NanoControl.

        Returns
            str: Response message from the device due to stop command.

        Raises:
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.
        """

        return self._send_command("stop")

    @tested
    def stopnack(self) -> None:
        """
        Stops any command currently running on the NanoControl but
        returns no acknowledgement and therefore has lower latency.

        This command sends the stop signal to the device without waiting
        for confirmation, providing faster response times in emergency
        situations where immediate stopping is more important than
        knowing if the command was received.

        Use this when:
        - Fast stop is required (emergency situations)
        - Response confirmation is not important
        - Minimising latency is critical

        Use regular stop() when:
        - You need confirmation the command was received
        - Error handling is important
        - Latency is not critical

        Returns:
            None: No return value since no acknowledgement is expected.

        Raises:
            NanoControlConnectionError: If serial communication fails
                during command transmission (not device response).
            ValueError: If command contains non-ASCII characters.
        """

        try:
            # Send command without waiting for response
            command_bytes = ("stopnack" + "\r").encode("ascii")

            # Clear input buffer and send command
            self.device_serial.flushInput()
            self.device_serial.write(command_bytes)
            self.device_serial.flush()

        except UnicodeEncodeError as e:
            raise ValueError(
                "NanoControl.stopnack: Command contains non-ASCII "
                f"character. {e}"
            ) from e

        except (serial.SerialException, OSError, IOError) as e:
            raise NanoControlConnectionError(
                f"NanoControl.stopnack: Serial error during transmission: {e}"
            ) from e

    @tested
    def beep(self, frequency: int, duration_ms: int) -> str:
        """
        Produces an audible tone from the NanoControl device for a
        specified frequency and duration.

        Sends the 'beep' command to the NanoControl, causing it to emit
        a sound at the given frequency for the specified duration.

        Args:
            frequency (int): Tone frequency in Hertz. Must be a positive
                integer.
            duration_ms (int): Tone duration in milliseconds. Must be
                a positive integer.

        Returns:
            str: Response message from the NanoControl if the command is
                acknowledged successfully.

        Raises:
            ValueError: If either `frequency` or `duration_ms` is not
                a positive integer.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.
        """

        # Validate frequency
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError(
                "NanoControl.beep: frequency must be a positive integer."
            )

        # Validate duration
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError(
                "NanoControl.beep: duration_ms must be a positive integer."
            )

        # Construct and send the beep command.
        return self._send_command(f"beep {frequency} {duration_ms}")

    @tested
    def set_motor_drive_frequencies(self, frequencies: List[int]) -> str:
        """
        Set the motor drive frequencies for joints A to D of the
        NanoControl device.

        Each joint's stepper motor is driven at a specific frequency
        which affects the stepping rate and movement characteristics.

        Args:
            frequencies (List[int]): A list of 4 integer frequency
                values for joints A, B, C, and D respectively. Each
                value must be within the valid frequency range.

        Returns:
            str: Response message from the device confirming the
                frequency setting.

        Raises:
            ValueError: If the input is not a list of 4 integers, or
                if any frequency is outside the valid frequency range.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Note:
            The set frequencies persist until overwritten by another
            set of frequencies or the device is powered off.
        """

        # Validate input type and length
        if not isinstance(frequencies, list) or len(frequencies) != len(
            self.JOINT_NAMES
        ):
            raise ValueError(
                f"NanoControl.set_motor_drive_frequencies: frequencies "
                f"must be a list of {len(self.JOINT_NAMES)} integers."
            )

        for i, freq in enumerate(frequencies):
            # Explicitly reject booleans first, then check for int type
            if isinstance(freq, bool) or not isinstance(freq, int):
                raise ValueError(
                    f"NanoControl.set_motor_drive_frequencies: frequency "
                    f"for joint {self.JOINT_NAMES[i]} must be an integer, "
                    f"got {type(freq)}."
                )
            if not self.MIN_FREQUENCY <= freq <= self.MAX_FREQUENCY:
                raise ValueError(
                    f"NanoControl.set_motor_drive_frequencies: frequency "
                    f"for joint {self.JOINT_NAMES[i]} must be between "
                    f"{self.MIN_FREQUENCY} and {self.MAX_FREQUENCY} Hz, "
                    f"got {freq}."
                )

        # Construct the command
        command = f"frequency {' '.join(map(str, frequencies))}"

        # Send command to device
        return self._send_command(command)

    @tested
    def get_version(self) -> str:
        """
        Queries the NanoControl device for its firmware version and
        build date.

        Returns:
            str: Firmware version and build date string.

        Raises:
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.
        """

        # Since _send_command raises on errors, response should be 'o'
        # or 'i' status message.
        return self._send_command("version")

    @tested
    def set_fine_with_coarse_all(self, status: bool) -> str:
        """
        Enable or disable the mode that allows smooth coarse steps when
        fine step limits have been reached for all channels.

        When enabled, the device will automatically perform coarse
        movements to continue motion smoothly when fine positioning has
        reached its limits. When disabled, motion will stop at fine step
        limits.

        Args:
            status (bool): True to enable fine-with-coarse mode,
                        False to disable it.

        Returns:
            str: Response message from the device confirming the mode
                change.

        Raises:
            ValueError: If status is not a boolean.
            NanoControlConnectionError: If serial communication fails.
            NanoControlCommandError: If the device returns an error,
                times out, or sends an unknown response.

        Note:
            This setting persists until overwritten by another call of
            the function or until the device is powered off.
        """

        # Validate input
        if not isinstance(status, bool):
            raise ValueError(
                "NanoControl.set_fine_with_coarse_all: status "
                "must be a boolean."
            )

        # Convert boolean to command parameter (1 for True, 0 for False)
        mode_value = int(status)

        return self._send_command(
            f"finewithcoarse {mode_value} {mode_value} "
            f"{mode_value} {mode_value}"
        )
