import os
import numpy as np

# ==== CHỈ SỬA CHỖ NÀY NẾU CẦN ====
# Thư mục output LLIRL của bạn
llirl_output_path = "llirl_sumo/output/120p4k_ultimate_test2"
rewards_filename = "rews_llirl.npy"
# ==================================

def main():
    rewards_file = os.path.join(llirl_output_path, rewards_filename)
    rewards_file_tmp = rewards_file + ".tmp.npy"

    if not os.path.exists(rewards_file):
        if os.path.exists(rewards_file_tmp):
            rewards_file = rewards_file_tmp
        else:
            print(f"[ERR] Không tìm thấy file reward: {rewards_file} hoặc {rewards_file_tmp}")
            return

    print(f"Đang load file: {rewards_file}")
    rewards = np.load(rewards_file)

    print("\n=== THÔNG TIN CƠ BẢN ===")
    print("Shape:", rewards.shape)
    print("Số chiều:", rewards.ndim)

    if rewards.ndim == 1:
        print("\nMảng 1 chiều, giá trị reward theo period/episode:")
        print(rewards)
        return

    # Nếu là 2D như (5, 10)
    num_periods, num_eps = rewards.shape
    print(f"\nMảng 2 chiều: {num_periods} periods x {num_eps} episodes/iters")

    print("\n=== KIỂM TRA TỪNG PERIOD CÓ BỊ LẶP HAY KHÔNG ===")
    for p in range(num_periods):
        row = rewards[p]
        unique_vals = np.unique(row)
        print(f"\nPeriod {p}:")
        print("  Giá trị trong period:", row)
        print("  Số giá trị khác nhau:", len(unique_vals))
        print("  Các giá trị unique:", unique_vals)
        if len(unique_vals) == 1:
            print("  👉 Period này TẤT CẢ 10 PHẦN TỬ ĐỀU GIỐNG NHAU.")
        else:
            print("  ✅ Period này có nhiều reward khác nhau.")

    print("\n=== GỢI Ý NGUYÊN NHÂN ===")
    print("- Nếu mỗi period chỉ có 1 giá trị unique → khả năng cao bạn gán như: rews_llirl[p] = period_reward")
    print("- Nếu muốn lưu reward từng episode: cần gán dạng rews_llirl[p, e] = ep_reward trong vòng for")

if __name__ == "__main__":
    main()
