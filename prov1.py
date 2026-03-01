    import serial
    import struct
    import threading
    import time
    from datetime import datetime
    import numpy as np
    import math
    from crc8 import crc8
    from crc_utils import crc_table

    np.set_printoptions(suppress=True)

    # ----------------- LIDAR CLASS -----------------
    class LIDAR:
        def __init__(self, serial_port, baudrate):
            self.PACKET_LENGTH = 49
            self.POINT_PER_PACK = 12
            self.serial_conn = serial.Serial(serial_port, baudrate=baudrate, timeout=1)

        def calculate_crc8(self, data):
            crc = 0x00
            for byte in data:
                crc = crc_table[(crc ^ byte) & 0xFF]
            return crc

        def parse_packet(self, packet):
            if len(packet) != self.PACKET_LENGTH:
                return

            if packet[0] != 0x54 or packet[1] != 0x2C:
                return

            received_crc = packet[self.PACKET_LENGTH - 3]
            calculated_crc = self.calculate_crc8(packet[: self.PACKET_LENGTH - 3])
            if received_crc != calculated_crc:
                return

            header, ver_len, speed, start_angle = struct.unpack("<BBHH", packet[:6])
            end_angle, timestamp = struct.unpack("<HH", packet[42:46])

            distances = np.zeros((self.POINT_PER_PACK,))
            intensities = np.zeros((self.POINT_PER_PACK,))
            angles = np.zeros((self.POINT_PER_PACK,))

            start_angle = (start_angle % 36000) / 100.0
            end_angle = (end_angle % 36000) / 100.0

            angle_diff = (end_angle - start_angle + 360.0) % 360.0
            angle_increment = angle_diff / 11  # 12 points, 11 intervals

            offset = 6
            for i in range(self.POINT_PER_PACK):
                distance, intensity = struct.unpack("<HB", packet[offset: offset + 3])
                distances[i] = distance
                intensities[i] = intensity
                angles[i] = (start_angle + i * angle_increment) % 360.0
                offset += 3

            return angles, distances

        def read_lidar_data(self):
            packet = self.serial_conn.read_until(b"\x54\x2C")
            if len(packet) != (self.PACKET_LENGTH - 2):
                return
            packet = bytes(b"\x54\x2C") + packet
            if len(packet) == self.PACKET_LENGTH:
                if polar_coord := self.parse_packet(packet):
                    info = np.stack(polar_coord).T
                    return info

        def close_serial_connection(self):
            self.serial_conn.close()


    # ----------------- UART Communication -----------------
    def read_packet(ser):
        chunk = ser.read_until(b"\x7E")
        if len(chunk) != 17:
            return
        values = struct.unpack("<ffffx", chunk)
        return values


    # ----------------- MOTION CONTROL -----------------
    def motion_control(lidar_info, side="right", max_speed=5, safe_dist=0.5):
        """
        side: 'right' или 'left' - по какой стене держаться
        max_speed: максимальная скорость колес
        safe_dist: безопасное расстояние до стены вперед
        """
        theta = lidar_info["theta"]
        length = lidar_info["length"]

        if theta is None or length is None:
            return 0, 0  # нет данных

        # Преобразуем в массив numpy
        theta = np.array(theta)
        length = np.array(length)

        forward_idx = (theta > -15*math.pi/180) & (theta < 15*math.pi/180)
        left_idx = (theta > 75*math.pi/180) & (theta < 105*math.pi/180)
        right_idx = (theta > -105*math.pi/180) & (theta < -75*math.pi/180)

        dist_forward = np.min(length[forward_idx]) if np.any(forward_idx) else 10
        dist_left = np.min(length[left_idx]) if np.any(left_idx) else 10
        dist_right = np.min(length[right_idx]) if np.any(right_idx) else 10

        wl_star = max_speed
        wr_star = max_speed

        # Держимся выбранной стороны
        if side == "right":
            if dist_right > safe_dist:
                wl_star = max_speed
                wr_star = max_speed * 0.8
            else:
                wl_star = max_speed * 0.8
                wr_star = max_speed
        else:
            if dist_left > safe_dist:
                wl_star = max_speed * 0.8
                wr_star = max_speed
            else:
                wl_star = max_speed
                wr_star = max_speed * 0.8

        # Поворот при стене впереди
        if dist_forward < safe_dist:
            if side == "right":
                wl_star = max_speed * 0.8
                wr_star = -max_speed * 0.8
            else:
                wl_star = -max_speed * 0.8
                wr_star = max_speed * 0.8

        return wl_star, wr_star


    # ----------------- ROBOT PROCESS -----------------
    def robot_process(info: dict):
        T = 1/50  # 50 Hz
        with serial.Serial(
            "/dev/ttyAMA0", 115200, bytesize=8,
            stopbits=serial.STOPBITS_ONE, timeout=1000
        ) as ser:
            while info["state"]:
                t1 = time.time()

                values = read_packet(ser)
                if values is None:
                    continue

                info["th_l"], info["th_r"], info["w_l"], info["w_r"] = values
                info["time"] = time.time()

                buf = struct.pack("<ff", info["wl_star"], info["wr_star"])
                crc = crc8()
                crc.update(buf)
                buf = b"\x7E" + buf + crc.digest()
                ser.write(buf)
                ser.flush()

                dtime = T - (time.time() - t1)
                if dtime > 0:
                    time.sleep(dtime)


    # ----------------- LIDAR PROCESS -----------------
    def lidar_process(info: dict):
        lidar = LIDAR(serial_port="/dev/ttyUSB0", baudrate=230400)
        prev_end_angle = 0
        end_angle = 0
        laser_scan = []
        while info["state"]:
            data = lidar.read_lidar_data()
            if data is None:
                continue

            prev_end_angle = end_angle
            end_angle = data[-1, 0]

            if not (prev_end_angle > 180 and end_angle < 180):
                laser_scan.append(data)
                continue

            laser_scan.append(data)
            scan_array = np.vstack(laser_scan)
            laser_scan = []

            th = scan_array[:, 0] * np.pi / 180
            l = scan_array[:, 1] / 1000

            info["theta"] = th
            info["length"] = l
            info["time"] = time.time()


    # ----------------- MAIN -----------------
    if __name__ == "__main__":
        robot_info = {
            "state": True, "th_l": None, "th_r": None,
            "w_l": None, "w_r": None,
            "wl_star": 0, "wr_star": 0, "time": time.time()
        }
        lidar_info = {"state": True, "theta": None, "length": None, "time": time.time()}

        robot_thread = threading.Thread(target=robot_process, args=(robot_info,))
        lidar_thread = threading.Thread(target=lidar_process, args=(lidar_info,))

        robot_thread.start()
        lidar_thread.start()

        print("Waiting for data...")
        while robot_info["th_l"] is None or lidar_info["theta"] is None:
            time.sleep(0.1)

        print("Start moving")
        try:
            T = 1/50
            while True:
                t1 = time.time()

                wl_star, wr_star = motion_control(lidar_info, side="left", max_speed=5, safe_dist=0.5)
                robot_info["wl_star"] = wl_star
                robot_info["wr_star"] = wr_star

                print("Wheel angles:", robot_info["th_l"], robot_info["th_r"])
                print("Wheel commands:", wl_star, wr_star)

                dtime = T - (time.time() - t1)
                if dtime > 0:
                    time.sleep(dtime)

        except KeyboardInterrupt:
            print("\nStopping robot")
            robot_info["wl_star"] = 0
            robot_info["wr_star"] = 0
            time.sleep(1)
            robot_info["state"] = False
            lidar_info["state"] = False

        finally:
            robot_thread.join()
            lidar_thread.join()
            print("All threads finished")
