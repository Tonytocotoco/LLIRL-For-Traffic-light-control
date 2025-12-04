# controller/env.py
import traci
from .strategies import PHASE_NS, PHASE_Y_NS, PHASE_EW, PHASE_Y_EW, KEEP, SWITCH

class TLSEnv:
    def __init__(self, cfg="run.sumocfg", lane_groups=None,
                 step_len=1.0, reward_type="neg_wait",
                 tripinfo=None, summary=None, vehroute=None,
                 teleport_time=None, extra_sumo_args=None):
        """
        Khởi tạo môi trường SUMO + TraCI.
        Các tham số chính:
        - cfg: đường dẫn file cấu hình .sumocfg
        - lane_groups: đối tượng LaneGroups (gom nhóm lane/edge)
        - step_len: độ dài bước mô phỏng (giây)
        - reward_type: loại hàm thưởng
        - tripinfo/summary/vehroute: đường dẫn output SUMO
        - teleport_time: thời gian teleport tối đa (giây), None = mặc định, -1 = tắt teleport
        - extra_sumo_args: list tham số SUMO bổ sung (nếu cần)
        """
        self.cfg = cfg
        self.lg = lane_groups
        self.step_len = step_len
        self.reward_type = reward_type
        self.tripinfo = tripinfo
        self.summary = summary
        self.vehroute = vehroute
        self.teleport_time = teleport_time
        self.extra_sumo_args = extra_sumo_args or []

        # Biến nội bộ
        self.tls = None
        self.phase = None
        self.switch_count = 0


    def reset(self):
        """
        Khởi động SUMO, kết nối TraCI, trả về state ban đầu.
        """
        import traci

        # Cấu hình lệnh SUMO
        cmd = ["sumo", "-c", self.cfg, "--step-length", str(self.step_len)]

        # Xuất file thống kê (nếu được cấu hình)
        if self.tripinfo:
            cmd += ["--tripinfo-output", self.tripinfo]
        if self.summary:
            cmd += ["--summary-output", self.summary]
        if self.vehroute:
            cmd += ["--vehroute-output", self.vehroute]

        # Thêm time-to-teleport nếu người dùng cấu hình
        if self.teleport_time is not None:
            cmd += ["--time-to-teleport", str(self.teleport_time)]

        # Thêm các tham số SUMO bổ sung (nếu có)
        cmd += self.extra_sumo_args

        # Khởi động mô phỏng
        traci.start(cmd)

        # Lấy ID đèn đầu tiên trong mạng
        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            traci.close()
            raise RuntimeError("No traffic light found in network.")
        self.tls = tls_list[0]

        # Đặt pha ban đầu là NS xanh (giống strategies)
        from .strategies import PHASE_NS
        traci.trafficlight.setPhase(self.tls, PHASE_NS)
        self.phase = PHASE_NS
        self.switch_count = 0

        # Trả về state ban đầu
        return self._get_state()
    def step(self, action):
        # áp dụng action: SWITCH nghĩa là chuyển sang pha vàng tương ứng (env lo chuỗi vàng->xanh)
        if action == SWITCH:
            if self.phase == PHASE_NS:    self._set_phase(PHASE_Y_NS)
            elif self.phase == PHASE_EW:  self._set_phase(PHASE_Y_EW)
            elif self.phase == PHASE_Y_NS: self._set_phase(PHASE_EW)
            elif self.phase == PHASE_Y_EW: self._set_phase(PHASE_NS)
            self.switch_count += 1

        traci.simulationStep()

        # nếu đang ở pha vàng, tự chuyển ngay sang xanh tiếp theo (để strategy đơn giản)
        p = traci.trafficlight.getPhase(self.tls)
        if p == PHASE_Y_NS:
            self._set_phase(PHASE_EW)
        elif p == PHASE_Y_EW:
            self._set_phase(PHASE_NS)

        state = self._get_state()
        reward = self._reward(state)
        done = traci.simulation.getMinExpectedNumber() == 0
        info = {"phase": self.phase, "switch_count": self.switch_count}
        return state, reward, done, info

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    # ==== helpers ====
    def _set_phase(self, phase):
        traci.trafficlight.setPhase(self.tls, phase)
        self.phase = phase

    def _get_state(self):
        # queues & vehicles
        q_ns = self.lg.halting_on(traci, self.lg.IN_NS)
        q_ew = self.lg.halting_on(traci, self.lg.IN_EW)
        v_ns = self.lg.veh_on(traci, self.lg.IN_NS)
        v_ew = self.lg.veh_on(traci, self.lg.IN_EW)
        # simple pressures
        p_ns = q_ns - self.lg.veh_on(traci, self.lg.OUT_NS)
        p_ew = q_ew - self.lg.veh_on(traci, self.lg.OUT_EW)
        return {"q_ns": q_ns, "q_ew": q_ew, "v_ns": v_ns, "v_ew": v_ew,
                "p_ns": p_ns, "p_ew": p_ew, "phase": self.phase}

    def _reward(self, s):
        if self.reward_type == "neg_wait":
            return -(s["q_ns"] + s["q_ew"])
        if self.reward_type == "neg_queue_sq":
            q = s["q_ns"] + s["q_ew"]; return -(q*q)
        if self.reward_type == "pressure":
            return s["p_ns"] + s["p_ew"]
        return -(s["q_ns"] + s["q_ew"])
