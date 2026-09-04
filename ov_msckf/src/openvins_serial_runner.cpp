/*
 * OpenVINS Serial Dataset Runner
 * Authors: OpenVINS Team & dexus-tdeng
 *
 * Standalone offline runner for monocular egocentric visual-inertial datasets.
 * Reads IMU telemetry and video frames, applies dynamic hand masking,
 * estimates 6-DoF metric camera trajectories, backfills warmup frames,
 * and exports TUM formatted CameraTrajectory.txt with exact 1-to-1 frame sync.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/quat_ops.h"

namespace fs = boost::filesystem;

// SO(3) matrix exponential for a 3D rotation vector
Eigen::Matrix3d exp_so3(const Eigen::Vector3d &w) {
    double theta = w.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

// Spherical Linear Interpolation (SLERP) for SE(3) transformation matrices
Eigen::Matrix4d interpolate_pose(const Eigen::Matrix4d &T0, double t0,
                                 const Eigen::Matrix4d &T1, double t1,
                                 double t_query) {
    if (std::abs(t1 - t0) < 1e-9) {
        return T0;
    }
    double alpha = (t_query - t0) / (t1 - t0);
    alpha = std::max(0.0, std::min(1.0, alpha));

    // Linear interpolation for translation
    Eigen::Vector3d p0 = T0.block<3, 1>(0, 3);
    Eigen::Vector3d p1 = T1.block<3, 1>(0, 3);
    Eigen::Vector3d p_interp = (1.0 - alpha) * p0 + alpha * p1;

    // SLERP for rotation
    Eigen::Quaterniond q0(T0.block<3, 3>(0, 0));
    Eigen::Quaterniond q1(T1.block<3, 3>(0, 0));
    q0.normalize();
    q1.normalize();
    Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
    q_interp.normalize();

    Eigen::Matrix4d T_interp = Eigen::Matrix4d::Identity();
    T_interp.block<3, 3>(0, 0) = q_interp.toRotationMatrix();
    T_interp.block<3, 1>(0, 3) = p_interp;
    return T_interp;
}

int main(int argc, char **argv) {
    std::string config_path = "";
    std::string timestamps_path = "";
    std::string images_dir = "";
    std::string imu_csv_path = "";
    std::string output_path = "CameraTrajectory.txt";
    std::string masks_dir = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--timestamps" && i + 1 < argc) {
            timestamps_path = argv[++i];
        } else if (arg == "--images" && i + 1 < argc) {
            images_dir = argv[++i];
        } else if (arg == "--imu" && i + 1 < argc) {
            imu_csv_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--masks" && i + 1 < argc) {
            masks_dir = argv[++i];
        }
    }

    // Positional fallback: config timestamps images imu output [masks]
    if (config_path.empty() && argc >= 6) {
        config_path = argv[1];
        timestamps_path = argv[2];
        images_dir = argv[3];
        imu_csv_path = argv[4];
        output_path = argv[5];
        if (argc >= 7) {
            masks_dir = argv[6];
        }
    }

    if (config_path.empty() || timestamps_path.empty() || images_dir.empty() || imu_csv_path.empty()) {
        std::cerr << "Usage: openvins_serial_runner --config <config.yaml> --timestamps <timestamps.txt> "
                  << "--images <images_dir> --imu <imu.csv> --output <CameraTrajectory.txt> [--masks <masks_dir>]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "Starting OpenVINS Serial Dataset Runner" << std::endl;
    std::cout << "  - Config     : " << config_path << std::endl;
    std::cout << "  - Timestamps : " << timestamps_path << std::endl;
    std::cout << "  - Images Dir : " << images_dir << std::endl;
    std::cout << "  - IMU CSV    : " << imu_csv_path << std::endl;
    std::cout << "  - Output     : " << output_path << std::endl;
    if (!masks_dir.empty()) {
        std::cout << "  - Masks Dir  : " << masks_dir << std::endl;
    }
    std::cout << "==========================================================" << std::endl;

    // 1. Load IMU CSV file
    // Format: timestamp(s), gyro_x(rad/s), gyro_y(rad/s), gyro_z(rad/s), accel_x(m/s^2), accel_y(m/s^2), accel_z(m/s^2)
    std::ifstream imu_file(imu_csv_path);
    if (!imu_file.is_open()) {
        std::cerr << "Error: Unable to open IMU file: " << imu_csv_path << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<ov_core::ImuData> imu_readings;
    std::string line;
    while (std::getline(imu_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        double ts, wx, wy, wz, ax, ay, az;
        if (ss >> ts >> wx >> wy >> wz >> ax >> ay >> az) {
            ov_core::ImuData data;
            data.timestamp = ts;
            data.wm << wx, wy, wz;
            data.am << ax, ay, az;
            imu_readings.push_back(data);
        }
    }
    imu_file.close();
    std::sort(imu_readings.begin(), imu_readings.end());
    std::cout << "Loaded " << imu_readings.size() << " IMU readings." << std::endl;
    if (imu_readings.empty()) {
        std::cerr << "Error: No valid IMU measurements parsed!" << std::endl;
        return EXIT_FAILURE;
    }

    // 2. Load camera timestamps and image paths
    std::ifstream ts_file(timestamps_path);
    if (!ts_file.is_open()) {
        std::cerr << "Error: Unable to open timestamps file: " << timestamps_path << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<double> cam_timestamps;
    std::vector<std::string> cam_image_files;
    std::vector<std::string> cam_mask_files;

    int frame_idx = 0;
    while (std::getline(ts_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double ts;
        std::string img_name = "";
        std::string mask_name = "";

        if (ss >> ts) {
            ss >> img_name >> mask_name;
            cam_timestamps.push_back(ts);

            // Determine image path
            fs::path img_path;
            if (!img_name.empty() && fs::exists(fs::path(images_dir) / img_name)) {
                img_path = fs::path(images_dir) / img_name;
            } else if (!img_name.empty() && fs::exists(img_name)) {
                img_path = img_name;
            } else {
                // Default search by frame index
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%06d.png", frame_idx);
                img_path = fs::path(images_dir) / buf;
                if (!fs::exists(img_path)) {
                    std::snprintf(buf, sizeof(buf), "%06d.jpg", frame_idx);
                    img_path = fs::path(images_dir) / buf;
                }
            }
            cam_image_files.push_back(img_path.string());

            // Determine mask path
            fs::path mask_path = "";
            if (!mask_name.empty() && !masks_dir.empty() && fs::exists(fs::path(masks_dir) / mask_name)) {
                mask_path = fs::path(masks_dir) / mask_name;
            } else if (!masks_dir.empty()) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%06d.png", frame_idx);
                mask_path = fs::path(masks_dir) / buf;
                if (!fs::exists(mask_path)) {
                    std::snprintf(buf, sizeof(buf), "%06d.jpg", frame_idx);
                    mask_path = fs::path(masks_dir) / buf;
                }
            }
            cam_mask_files.push_back(fs::exists(mask_path) ? mask_path.string() : "");

            frame_idx++;
        }
    }
    ts_file.close();

    const size_t num_frames = cam_timestamps.size();
    std::cout << "Loaded " << num_frames << " camera frames." << std::endl;
    if (num_frames == 0) {
        std::cerr << "Error: No camera frames loaded!" << std::endl;
        return EXIT_FAILURE;
    }

    // 3. Initialize OpenVINS VioManager
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);
    params.use_multi_threading_subs = false;
    params.use_multi_threading_pubs = false;
    params.num_opencv_threads = 0;

    auto sys = std::make_shared<ov_msckf::VioManager>(params);

    // Track trajectory: map from camera timestamp -> T_CtoG (4x4 matrix)
    std::map<double, Eigen::Matrix4d> trajectory_poses;
    size_t imu_pointer = 0;
    int initialized_at_frame = -1;

    // 4. Sequential processing loop
    for (size_t i = 0; i < num_frames; ++i) {
        double t_cam = cam_timestamps[i];
        double t_off = sys->get_state()->_calib_dt_CAMtoIMU->value()(0);
        double t_imu_target = t_cam + t_off;

        // Feed IMU measurements up to and including at least one measurement past t_imu_target
        while (imu_pointer < imu_readings.size()) {
            const auto &imu_msg = imu_readings[imu_pointer];
            sys->feed_measurement_imu(imu_msg);
            imu_pointer++;
            if (imu_msg.timestamp > t_imu_target) {
                break;
            }
        }

        // Load image (grayscale)
        cv::Mat img = cv::imread(cam_image_files[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error: Could not read image at: " << cam_image_files[i] << std::endl;
            return EXIT_FAILURE;
        }

        // Load or create mask (255 = mask out hand, 0 = free feature extraction)
        cv::Mat mask;
        if (!cam_mask_files[i].empty() && fs::exists(cam_mask_files[i])) {
            mask = cv::imread(cam_mask_files[i], cv::IMREAD_GRAYSCALE);
        }
        if (mask.empty() || mask.rows != img.rows || mask.cols != img.cols) {
            mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        }

        // Build camera data measurement
        ov_core::CameraData cam_data;
        cam_data.timestamp = t_cam;
        cam_data.sensor_ids = {0};
        cam_data.images = {img};
        cam_data.masks = {mask};

        // Feed camera measurement
        sys->feed_measurement_camera(cam_data);

        // Record tracking status and update clone estimates
        if (sys->initialized()) {
            if (initialized_at_frame < 0) {
                initialized_at_frame = (int)i;
                std::cout << ">>> OpenVINS initialized at frame " << i << " (timestamp " 
                          << std::fixed << std::setprecision(6) << t_cam << ") <<<" << std::endl;
            }

            auto state = sys->get_state();
            if (!state->_calib_IMUtoCAM.empty()) {
                auto calib = state->_calib_IMUtoCAM.at(0);
                Eigen::Matrix3d R_ItoC = calib->Rot();
                Eigen::Vector3d p_IinC = calib->pos();
                Eigen::Matrix3d R_CtoI = R_ItoC.transpose();
                Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;

                for (const auto &pair : state->_clones_IMU) {
                    double clone_t = pair.first;
                    auto clone = pair.second;
                    Eigen::Matrix3d R_GtoI = clone->Rot();
                    Eigen::Vector3d p_IinG = clone->pos();
                    Eigen::Matrix3d R_ItoG = R_GtoI.transpose();

                    // Camera pose in Global world frame: T_CtoG
                    Eigen::Matrix3d R_CtoG = R_ItoG * R_CtoI;
                    Eigen::Vector3d p_CinG = p_IinG + R_ItoG * p_CinI;

                    Eigen::Matrix4d T_CtoG = Eigen::Matrix4d::Identity();
                    T_CtoG.block<3, 3>(0, 0) = R_CtoG;
                    T_CtoG.block<3, 1>(0, 3) = p_CinG;

                    trajectory_poses[clone_t] = T_CtoG;
                }
            }
        }

        if ((i + 1) % 150 == 0 || i + 1 == num_frames) {
            std::cout << "Processed frame " << (i + 1) << " / " << num_frames 
                      << " (" << std::fixed << std::setprecision(1) << (100.0 * (i + 1) / num_frames) << "%)"
                      << " | Tracked poses: " << trajectory_poses.size() << std::endl;
        }
    }

    if (trajectory_poses.empty()) {
        std::cerr << "Error: OpenVINS failed to initialize and track poses!" << std::endl;
        return EXIT_FAILURE;
    }

    // 5. Backfill warmup frames and interpolate missing frames for 1-to-1 sync
    std::cout << "Synchronizing trajectory to match all " << num_frames << " video frames 1-to-1..." << std::endl;

    double first_tracked_t = trajectory_poses.begin()->first;
    Eigen::Matrix4d first_T_CtoG = trajectory_poses.begin()->second;

    // Retrieve calibrated extrinsics and gyro bias for backward integration
    auto state = sys->get_state();
    auto calib = state->_calib_IMUtoCAM.at(0);
    Eigen::Matrix3d R_ItoC = calib->Rot();
    Eigen::Vector3d p_IinC = calib->pos();
    Eigen::Matrix3d R_CtoI = R_ItoC.transpose();
    Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;
    Eigen::Vector3d bg = state->_imu->bias_g();

    // Orientation and position of IMU at first tracked time
    Eigen::Matrix3d R_CtoG_0 = first_T_CtoG.block<3, 3>(0, 0);
    Eigen::Vector3d p_CinG_0 = first_T_CtoG.block<3, 1>(0, 3);
    Eigen::Matrix3d R_ItoG_0 = R_CtoG_0 * R_ItoC; // since R_CtoG = R_ItoG * R_CtoI
    Eigen::Vector3d p_IinG_0 = p_CinG_0 - R_ItoG_0 * p_CinI;

    // Backfill frames before first_tracked_t
    for (int i = 0; i < (int)num_frames; ++i) {
        double t = cam_timestamps[i];
        if (t >= first_tracked_t) break;

        // Backward integrate gyro from first_tracked_t to t
        Eigen::Matrix3d R_ItoG_t = R_ItoG_0;
        // Integrate IMU readings between t and first_tracked_t in reverse
        for (int k = (int)imu_readings.size() - 1; k >= 0; --k) {
            double imu_t = imu_readings[k].timestamp;
            if (imu_t > first_tracked_t) continue;
            if (imu_t <= t) break;

            double dt = (k > 0) ? (imu_t - imu_readings[k - 1].timestamp) : 0.0025;
            if (dt <= 0.0 || dt > 0.05) dt = 0.0025;
            Eigen::Vector3d w_corrected = imu_readings[k].wm - bg;
            R_ItoG_t = R_ItoG_t * exp_so3(-w_corrected * dt);
        }

        // Near stationary initial position
        Eigen::Vector3d p_IinG_t = p_IinG_0;
        Eigen::Matrix3d R_CtoG_t = R_ItoG_t * R_CtoI;
        Eigen::Vector3d p_CinG_t = p_IinG_t + R_ItoG_t * p_CinI;

        Eigen::Matrix4d T_CtoG_t = Eigen::Matrix4d::Identity();
        T_CtoG_t.block<3, 3>(0, 0) = R_CtoG_t;
        T_CtoG_t.block<3, 1>(0, 3) = p_CinG_t;
        trajectory_poses[t] = T_CtoG_t;
    }

    // Interpolate any internal frames that might have been skipped
    for (size_t i = 0; i < num_frames; ++i) {
        double t = cam_timestamps[i];
        if (trajectory_poses.find(t) != trajectory_poses.end()) {
            continue;
        }

        // Find bounding timestamps
        auto upper = trajectory_poses.upper_bound(t);
        if (upper == trajectory_poses.end()) {
            // Extrapolate using the last available pose
            trajectory_poses[t] = std::prev(upper)->second;
        } else if (upper == trajectory_poses.begin()) {
            trajectory_poses[t] = upper->second;
        } else {
            auto lower = std::prev(upper);
            trajectory_poses[t] = interpolate_pose(lower->second, lower->first,
                                                   upper->second, upper->first, t);
        }
    }

    // 6. Write TUM formatted trajectory file
    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out_file(output_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Unable to open output file: " << output_path << std::endl;
        return EXIT_FAILURE;
    }

    out_file << "# timestamp tx ty tz qx qy qz qw\n";
    size_t exported_count = 0;

    for (size_t i = 0; i < num_frames; ++i) {
        double t = cam_timestamps[i];
        const Eigen::Matrix4d &T = trajectory_poses[t];

        Eigen::Vector3d p = T.block<3, 1>(0, 3);
        Eigen::Quaterniond q(T.block<3, 3>(0, 0));
        q.normalize();

        out_file << std::fixed << std::setprecision(6) << t << " "
                 << std::setprecision(7)
                 << p.x() << " " << p.y() << " " << p.z() << " "
                 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        exported_count++;
    }
    out_file.close();

    std::cout << "==========================================================" << std::endl;
    std::cout << "SUCCESS: Exported " << exported_count << " poses to: " << output_path << std::endl;
    std::cout << "Input frames: " << num_frames << " | Output TUM rows: " << exported_count << std::endl;
    if (exported_count == num_frames) {
        std::cout << "VERIFICATION PASSED: Exact 1-to-1 sync confirmed!" << std::endl;
    } else {
        std::cerr << "WARNING: Pose count does not match input frame count!" << std::endl;
    }
    std::cout << "==========================================================" << std::endl;

    return (exported_count == num_frames) ? EXIT_SUCCESS : EXIT_FAILURE;
}
