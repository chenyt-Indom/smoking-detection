"""
camera_scanner.py - 自动扫描本地摄像头 + 无线摄像头
支持 USB 摄像头、RTSP 网络摄像头、ONVIF 协议
"""
import cv2
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class CameraScanner:
    """自动发现可用摄像头"""

    @staticmethod
    def scan_usb(max_index=10):
        """扫描本地 USB 摄像头"""
        cameras = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        h, w = frame.shape[:2]
                        cameras.append({
                            "id": f"usb_{i}",
                            "name": f"USB摄像头 #{i}",
                            "source": i,
                            "type": "usb",
                            "resolution": f"{w}x{h}",
                        })
                    cap.release()
            except Exception:
                pass
        return cameras

    @staticmethod
    def scan_rtsp(ip_range="192.168.1", ports=None, timeout=2):
        """扫描局域网内 RTSP 摄像头"""
        if ports is None:
            ports = [554, 8554, 8080, 80]

        found = []
        common_paths = [
            "/stream1", "/stream", "/live", "/h264", "/video",
            "/cam/realmonitor", "/onvif/device_service",
            "/Streaming/Channels/1", "/h264/ch1/main",
        ]

        def check_ip(ip):
            for port in ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                try:
                    result = sock.connect_ex((ip, port))
                    if result == 0:
                        for path in common_paths[:3]:
                            rtsp_url = f"rtsp://{ip}:{port}{path}"
                            try:
                                cap = cv2.VideoCapture(rtsp_url)
                                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
                                if cap.isOpened():
                                    ret, frame = cap.read()
                                    if ret:
                                        h, w = frame.shape[:2]
                                        found.append({
                                            "id": f"rtsp_{ip.replace('.', '_')}",
                                            "name": f"RTSP {ip}:{port}",
                                            "source": rtsp_url,
                                            "type": "rtsp",
                                            "resolution": f"{w}x{h}",
                                        })
                                    cap.release()
                            except Exception:
                                pass
                except Exception:
                    pass
                finally:
                    sock.close()

        ips = [f"{ip_range}.{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(check_ip, ip): ip for ip in ips}
            for future in as_completed(futures):
                try:
                    future.result(timeout=timeout)
                except Exception:
                    pass

        return found

    @staticmethod
    def scan_all(rtsp_range="192.168.1"):
        """扫描所有摄像头"""
        all_cameras = []

        # USB 摄像头
        usb = CameraScanner.scan_usb()
        all_cameras.extend(usb)
        print(f"[CameraScanner] 发现 {len(usb)} 个USB摄像头")

        # RTSP 摄像头（快速扫描，仅检查常见端口的前5个IP）
        # 完整扫描太慢，这里做快速扫描
        quick_ips = [f"{rtsp_range}.{i}" for i in [1, 100, 101, 200, 254]]
        for ip in quick_ips:
            for port in [554, 8554]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                try:
                    if sock.connect_ex((ip, port)) == 0:
                        all_cameras.append({
                            "id": f"rtsp_{ip.replace('.', '_')}",
                            "name": f"RTSP {ip}:{port}",
                            "source": f"rtsp://{ip}:{port}/stream1",
                            "type": "rtsp",
                            "resolution": "unknown",
                            "enabled": True,
                        })
                except Exception:
                    pass
                finally:
                    sock.close()

        print(f"[CameraScanner] 共发现 {len(all_cameras)} 个摄像头")
        return all_cameras