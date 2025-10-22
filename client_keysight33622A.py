"""
Client-side proxy for keysight33622A that communicates with the TCP server
Provides the exact same interface as keysight33622A but sends commands over TCP
Uses persistent connection for better performance
"""

import socket
import json
import time
import threading


class ClientKeysight33622A:
    """
    Client proxy for keysight33622A that communicates with a TCP server
    Provides the same interface as the original class
    Uses persistent connection for better performance
    """
    
    def __init__(self, ipAddress: str = "", server_host: str = "localhost", 
                 server_port: int = 8888, timeout: float = 5.0, **kwargs):
        """
        :param ipAddress: Ignored - kept for compatibility with original interface
        :param server_host: Host where the Keysight server is running
        :param server_port: Port where the Keysight server is listening
        :param timeout: Timeout for socket operations
        :param kwargs: Additional arguments (ignored, kept for compatibility)
        """
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.high_level = 0  # Keep for compatibility
        self._connected = False
        self._socket = None
        self._lock = threading.Lock()  # Thread safety for socket operations
        
    def connect(self):
        """
        Connect to the server and maintain persistent connection
        """
        with self._lock:
            if self._connected:
                return True
                
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.server_host, self.server_port))
                
                self._connected = True
                print(f"Connected to Keysight server at {self.server_host}:{self.server_port}")
                return True
                
            except Exception as e:
                print(f"Failed to connect to Keysight server: {e}")
                self._connected = False
                if self._socket:
                    try:
                        self._socket.close()
                    except:
                        pass
                    self._socket = None
                return False
            
    def disconnect(self):
        """Close the persistent connection"""
        with self._lock:
            self._connected = False
            if self._socket:
                try:
                    self._socket.close()
                except:
                    pass
                self._socket = None
            return True
            
    def _send_request(self, method: str, *args, **kwargs):
        """Send a request to the server using persistent connection"""
        request = {
            'instrument': 'function_gen',  # Specify function generator
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        
        with self._lock:
            if not self._connected or not self._socket:
                raise RuntimeError("Not connected to server. Call connect() first.")
                
            try:
                # Send request
                request_json = json.dumps(request).encode('utf-8')
                self._socket.send(request_json)
                
                # Receive response
                response_data = self._socket.recv(4096)
                
                # Parse response
                response = json.loads(response_data.decode('utf-8'))
                
                if 'error' in response:
                    raise RuntimeError(f"Server error: {response['error']}")
                    
                return response.get('result')
                
            except socket.timeout:
                # Connection may be broken, mark as disconnected
                self._connected = False
                raise TimeoutError(f"Request to server timed out after {self.timeout} seconds")
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                # Connection was broken, mark as disconnected
                self._connected = False
                raise RuntimeError(f"Connection to server lost: {e}")
            except json.JSONDecodeError:
                raise RuntimeError("Invalid response from server")
            except Exception as e:
                # For other errors, try to reconnect next time
                self._connected = False
                raise RuntimeError(f"Communication error: {e}")
                
    def _send_request_with_retry(self, method: str, *args, **kwargs):
        """Send request with automatic reconnection on failure"""
        try:
            return self._send_request(method, *args, **kwargs)
        except (RuntimeError, TimeoutError) as e:
            # If connection failed, try to reconnect once
            if "Connection" in str(e) or "timeout" in str(e).lower():
                print("Connection lost, attempting to reconnect...")
                if self.connect():
                    return self._send_request(method, *args, **kwargs)
            raise

    # All the keysight33622A methods
    def init(self):
        """Initialize the instrument"""
        return self._send_request_with_retry('init')
        
    def reset(self):
        """Reset the instrument"""
        return self._send_request_with_retry('reset')
        
    def set_function(self, channel: int, function_type: str):
        """Set the function type for a specific channel"""
        return self._send_request_with_retry('set_function', channel, function_type)
        
    def set_pulse_width(self, channel: int, width: float):
        """Set the width of a pulse in seconds"""
        return self._send_request_with_retry('set_pulse_width', channel, width)
        
    def set_frequency(self, channel: int, freq: float):
        """Set the frequency of the output"""
        return self._send_request_with_retry('set_frequency', channel, freq)
        
    def set_amplitude(self, channel: int, amplitude: float):
        """Set the amplitude of the output"""
        return self._send_request_with_retry('set_amplitude', channel, amplitude)
        
    def set_offset(self, channel: int, offset: float):
        """Set the DC offset of the output"""
        return self._send_request_with_retry('set_offset', channel, offset)
        
    def set_phase(self, channel: int, phase: float):
        """Set the phase of the output in degrees"""
        return self._send_request_with_retry('set_phase', channel, phase)
        
    def apply_pulse(self, channel: int, freq: float, amplitude: float, offset: float):
        """Configure a pulse waveform with specified parameters"""
        return self._send_request_with_retry('apply_pulse', channel, freq, amplitude, offset)
        
    def get_output(self, channel: int):
        """Get the output state for a specific channel"""
        return self._send_request_with_retry('get_output', channel)
        
    def set_output(self, channel: int, state: int):
        """Turn the output on or off for a specific channel"""
        return self._send_request_with_retry('set_output', channel, state)
        
    def set_polarity(self, channel: int, polarity: str):
        """Set the polarity of the output"""
        return self._send_request_with_retry('set_polarity', channel, polarity)
        
    def phase_sync(self):
        """Synchronize the phase of all channels"""
        return self._send_request_with_retry('phase_sync')
        
    def enable_burst(self, channel: int, burst_count: int = 1):
        """Enable burst mode for a specific channel"""
        return self._send_request_with_retry('enable_burst', channel, burst_count)
        
    def disable_burst(self, channel: int):
        """Disable burst mode for a specific channel"""
        return self._send_request_with_retry('disable_burst', channel)
        
    def immediate_trigger(self, channel: int):
        """Trigger immediately"""
        return self._send_request_with_retry('immediate_trigger', channel)
        
    def trigger_with_polarity(self, channel: int, high_level: float, polarity: str):
        """Trigger with specific polarity"""
        return self._send_request_with_retry('trigger_with_polarity', channel, high_level, polarity)
        
    def filter_channel(self, phase: float, freq: float):
        """Legacy compatibility function for channel 1 settings"""
        return self._send_request_with_retry('filter_channel', phase, freq)
        
    def gating_channel(self, x):
        """Legacy compatibility function for channel 2 settings"""
        return self._send_request_with_retry('gating_channel', x)
        
    def channels_off(self):
        """Turn both channels off"""
        return self._send_request_with_retry('channels_off')
        
    def channels_on(self):
        """Turn both channels on"""
        return self._send_request_with_retry('channels_on')
        
    def phase_zero(self):
        """Set phase to zero for both channels"""
        return self._send_request_with_retry('phase_zero')
        
    def set_pulse_polarity(self, channel: int, polarity: str, high_level: float = 0):
        """Set the polarity of the pulse waveform"""
        return self._send_request_with_retry('set_pulse_polarity', channel, polarity, high_level)
        
    def setup_pulse(self, channel: int = 1, period: float = 0.5, width: float = 0.050, edge_time: str = "10000 ns"):
        """Set up a pulse waveform with specified parameters"""
        return self._send_request_with_retry('setup_pulse', channel, period, width, edge_time)
        
    def set_thermal_source_mode(self):
        """Set thermal source mode"""
        return self._send_request_with_retry('set_thermal_source_mode')
        
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()


# For backwards compatibility, create an alias
keysight33622A = ClientKeysight33622A


if __name__ == '__main__':
    # Test the client
    print("Testing Keysight 33622A client with persistent connection...")
    
    client = ClientKeysight33622A()
    
    try:
        print("Connecting to server...")
        client.connect()
        
        print("Testing function generator control...")
        print("Setting up channel 2...")
        client.set_output(2, 1)  # Turn on channel 2
        time.sleep(1)
        
        print("Setting channel 2 to pulse mode...")
        client.set_function(2, "PULS")
        time.sleep(1)
        
        print("Turning off channel 2...")
        client.set_output(2, 0)
        time.sleep(1)
        
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()
        print("Disconnected from server")