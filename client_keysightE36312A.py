"""
Client-side proxy for keysightE36312A that communicates with the TCP server
Provides the exact same interface as keysightE36312A but sends commands over TCP
Uses persistent connection for better performance
"""

import socket
import json
import time
import threading


class ClientKeysightE36312A:
    """
    Client proxy for keysightE36312A that communicates with a TCP server
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
            
    def init(self):
        """Initialize the instrument"""
        return self._send_request_with_retry('init')
        
    def reset(self):
        """Reset the instrument"""
        return self._send_request_with_retry('reset')
        
    def output_on(self, channel: int):
        """Turn on output for the specified channel"""
        return self._send_request_with_retry('output_on', channel)
        
    def output_off(self, channel: int):
        """Turn off output for the specified channel"""
        return self._send_request_with_retry('output_off', channel)
        
    def get_on_off(self, channel: int):
        """Get the on/off status of the specified channel"""
        return self._send_request_with_retry('get_on_off', channel)
        
    def getVoltage(self, channel: int):
        """Get voltage measurement for the specified channel"""
        result = self._send_request_with_retry('getVoltage', channel)
        return float(result) if result is not None else 0.0
        
    def getCurrent(self, channel: int):
        """Get current measurement for the specified channel"""
        result = self._send_request_with_retry('getCurrent', channel)
        return float(result) if result is not None else 0.0
        
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()


# For backwards compatibility, create an alias
keysightE36312A = ClientKeysightE36312A


if __name__ == '__main__':
    # Test the client
    print("Testing Keysight client with persistent connection...")
    
    client = ClientKeysightE36312A()
    
    try:
        print("Connecting to server...")
        client.connect()
        
        print("Testing multiple requests on same connection...")
        for i in range(5):
            print(f"Request {i+1}:")
            print(f"  Channel 3 status: {client.get_on_off(3)}")
            print(f"  Channel 3 voltage: {client.getVoltage(3)}")
            time.sleep(0.5)
        
        print("Testing channel control...")
        print("Turning on channel 3...")
        client.output_on(3)
        time.sleep(1)
        
        print(f"Status after ON: {client.get_on_off(3)}")
        print(f"Voltage after ON: {client.getVoltage(3)}")
        
        time.sleep(2)
        
        print("Turning off channel 3...")
        client.output_off(3)
        time.sleep(1)
        
        print(f"Status after OFF: {client.get_on_off(3)}")
        print(f"Voltage after OFF: {client.getVoltage(3)}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()
        print("Disconnected from server")