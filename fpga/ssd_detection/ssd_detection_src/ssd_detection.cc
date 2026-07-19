#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <sstream>
#include <dirent.h>
#include <stdio.h>
#include <sys/times.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include <arm_neon.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "paddle_api.h"  // NOLINT
#include "intelfpga.h"
#include "httplib.h"

using namespace paddle::lite_api;  // NOLINT
using namespace httplib;

#define input_shape 300  // 必须是2的整数倍

const PowerMode CPU_POWER_MODE = PowerMode::LITE_POWER_FULL;

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

// Object for storing all preprocessed data
struct ImageBlob {
  std::vector<float> mean_;
  std::vector<float> scale_;
};

std::vector<std::string> ReadLine(const std::string &path) {
  std::ifstream file(path);
  std::vector<std::string> line_vec;
  std::string line;
  if (file){
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      line_vec.push_back(line);
    }
  }
  return line_vec;
}

std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) res.push_back(item);
  }
  return res;
}

std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadLine(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], ' ');
    if (res.size() >= 2) dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleDetection lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "===End of PaddleDetection lite demo config===" << std::endl;
}

class Detector {
public:
  explicit Detector(const std::string& config_path) {
    config_ = LoadConfigTxt(config_path);
    PrintConfig(config_);

    rgb_img.create(cv::Size(input_shape, input_shape), CV_8UC3);
    img_float.create(cv::Size(input_shape, input_shape), CV_32FC3);

    std::string label_path = config_.at("label_path");
    class_names_ = ReadLine(label_path);

    loadModel();

    init_ImageData();
  }

  ~Detector() {
    fpga_release();
  }

  std::string GetClassName(int class_id) const {
    if (class_id >= 0 && class_id < (int)class_names_.size()) {
      return class_names_[class_id];
    }
    return "unknown";
  }

  void PrintResult(std::vector<Object> objects){
    for (auto obj: objects){
      std::cout << ", detect object: " << class_names_[obj.class_id] << ", score: " << obj.prob
                << ", location: x=" << obj.rec.x << ", y=" << obj.rec.y
                << ", width=" << obj.rec.width << ", height=" << obj.rec.height
                << std::endl;
    }
  }

  std::vector<Object> Detect(const cv::Mat& frame) {
    if (frame.empty()) {
      std::cerr << "[WARN] processFrame received empty frame\n";
      return {};
    }

    // input 0
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
    input_tensor0->Resize({ 1, 2 });
    auto* data0 = input_tensor0->mutable_data<float>();
    data0[0] = data0[1] = input_shape;

    // input1
    std::unique_ptr<Tensor> input_tensor1(std::move(predictor_->GetInput(1)));
    input_tensor1->Resize({ 1, 3, input_shape, input_shape});
    auto* data1 = input_tensor1->mutable_data<float>();
    if (frame.channels() == 1) {
      preprocessImgGray(frame, data1);
    } else {
      preprocessImg(frame, data1);
    }

    // input2
    std::unique_ptr<Tensor> input_tensor2(std::move(predictor_->GetInput(2)));
    input_tensor2->Resize({ 1, 2 });
    auto* data2 = input_tensor2->mutable_data<float>();
    data2[0] = static_cast<float>(input_shape) / static_cast<float>(frame.rows);
    data2[1] = static_cast<float>(input_shape) / static_cast<float>(frame.cols);

    predictor_->Run();

    std::unique_ptr<const Tensor> output_tensor(std::move(predictor_->GetOutput(0)));
    // outdata : [class_id, confidence, x_min, y_min, x_max, y_max]
    const float* outdata = output_tensor->data<float>();
    auto shape = output_tensor->shape();
    int64_t count = 1;
    for (auto& dim : shape) count *= dim;
    int num_detections = static_cast<int>(count / 6);
    float confidence_threshold = 0.45f;

    std::vector<Object> objects;
    for (int iw = 0; iw < num_detections; iw++) {
      if (confidence_threshold < outdata[1] && outdata[1] <= 1 && outdata[2] < outdata[4] && outdata[3] < outdata[5]) {
        Object obj;
        int x = static_cast<int>(outdata[2]);
        int y = static_cast<int>(outdata[3]);
        int w = static_cast<int>(outdata[4] - outdata[2]);
        int h = static_cast<int>(outdata[5] - outdata[3]);
        cv::Rect rec_clip = cv::Rect(x, y, w, h) & cv::Rect(0, 0, frame.cols, frame.rows);
        obj.class_id = static_cast<int>(outdata[0]);
        obj.prob = outdata[1];
        obj.rec = rec_clip;
        objects.push_back(obj);
      }
      outdata += 6;
    }
    return objects;
  }

  void visualize_result(std::vector<Object> objects, cv::Mat& image) {
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 1.f;
    int thickness = 1;
    for (auto obj: objects){
      auto rec_clip = obj.rec;
      std::cout << "image size: " << image.cols << ", " << image.rows
                << ", detect object: " << class_names_[obj.class_id] << ", score: " << obj.prob
                << ", location: x=" << rec_clip.x << ", y=" << rec_clip.y
                << ", width=" << rec_clip.width << ", height=" << rec_clip.height
                << std::endl;
      cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
      std::string str_prob = std::to_string(obj.prob);
      std::string text = std::string(class_names_[obj.class_id]) + ": " +
                          str_prob.substr(0, str_prob.find(".") + 4);
      cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
      float new_font_scale = rec_clip.width * 0.5f * font_scale / text_size.width;
      text_size = cv::getTextSize(text, font_face, new_font_scale, thickness, nullptr);
      cv::Point origin;
      origin.x = rec_clip.x + 3;
      origin.y = rec_clip.y + text_size.height + 3;
      cv::putText(image,
                  text,
                  origin,
                  font_face,
                  new_font_scale,
                  cv::Scalar(0, 255, 255),
                  thickness,
                  cv::LINE_AA);
    }
  }

private:
  std::map<std::string, std::string> config_;
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<std::string> class_names_;
  ImageBlob img_data;
  cv::Mat rgb_img;   // 复用缓冲区
  cv::Mat img_float;

  void loadModel() {
    std::string model_file = config_.at("model_file");
    int num_threads = std::stoi(config_.at("num_threads"));

    MobileConfig config;
    config.set_model_from_file(model_file);
    config.set_threads(num_threads);
    config.set_power_mode(CPU_POWER_MODE);

    predictor_ = CreatePaddlePredictor<MobileConfig>(config);
    if (!predictor_) {
      std::cerr << "[ERROR] Failed to create predictor\n";
      exit(1);
    }
  }

  void init_ImageData() {
    std::vector<float> mean_vals, scale_vals;
    std::vector<std::string> mean_str = split(config_.at("mean"), ',');
    std::vector<std::string> std_str = split(config_.at("std"), ',');
    std::transform(mean_str.begin(), mean_str.end(), back_inserter(mean_vals),
                    [](const std::string& s) { return std::stof(s); });
    std::transform(std_str.begin(), std_str.end(), back_inserter(scale_vals),
                    [](const std::string& s) { return std::stof(s); });
    if (mean_vals.size() != 3 || scale_vals.size() != 3) {
      std::cerr << "[ERROR] mean or scale size must equal to 3\n";
      exit(1);
    }
    img_data.mean_ = mean_vals;
    img_data.scale_ = scale_vals;
  }

  void preprocessImg(const cv::Mat& img, float* data) {
    // img is already 300×300 (Pi resizes before sending; detect_image_file resizes explicitly)
    img.copyTo(rgb_img);

    rgb_img.convertTo(img_float, CV_32FC3, 1);
    const float* dimg = reinterpret_cast<const float*>(img_float.data);
    neon_mean_scale(dimg, data);
  }

  // Single-channel input → 3-channel NCHW tensor (no cvtColor, no redundant channel copy)
  void preprocessImgGray(const cv::Mat& img, float* data) {
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC1, 1.0);
    const float* din = reinterpret_cast<const float*>(img_float.data);

    const int size = input_shape * input_shape;
    float* dout_c0 = data;
    float* dout_c1 = data + size;
    float* dout_c2 = data + size * 2;

    float mean = img_data.mean_[0];
    float scale = 1.f / img_data.scale_[0];
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vscale = vdupq_n_f32(scale);

    for (int i = 0; i < size; i += 4) {
      float32x4_t vin = vld1q_f32(din + i);
      float32x4_t vscaled = vmulq_f32(vsubq_f32(vin, vmean), vscale);
      vst1q_f32(dout_c0 + i, vscaled);
      vst1q_f32(dout_c1 + i, vscaled);
      vst1q_f32(dout_c2 + i, vscaled);
    }
  }

  // fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
  void neon_mean_scale(const float* din, float* dout) {
    float32x4_t vmean0 = vdupq_n_f32(img_data.mean_[0]);
    float32x4_t vmean1 = vdupq_n_f32(img_data.mean_[1]);
    float32x4_t vmean2 = vdupq_n_f32(img_data.mean_[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / img_data.scale_[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / img_data.scale_[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / img_data.scale_[2]);

    const int size = input_shape * input_shape;
    float* dout_c0 = dout;
    float* dout_c1 = dout + size;
    float* dout_c2 = dout + size * 2;

    for (int i = 0; i < size; i += 4) {
      float32x4x3_t vin3 = vld3q_f32(din);
      float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
      float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
      float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
      float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
      float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
      float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
      vst1q_f32(dout_c0, vs0);
      vst1q_f32(dout_c1, vs1);
      vst1q_f32(dout_c2, vs2);

      din += 12;
      dout_c0 += 4;
      dout_c1 += 4;
      dout_c2 += 4;
    }
  }
};

void detect_image_file(Detector& detector, std::string input_path){
  std::vector<std::string> img_paths;
  struct stat st;
  if (stat(input_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
    DIR* dp = opendir(input_path.c_str());
    dirent* entry;
    while (entry = readdir(dp)){
      if (entry->d_name[0] == '.') continue;
      img_paths.emplace_back(input_path + '/' + entry->d_name);
    }
    closedir(dp);
  }
  else img_paths.push_back(input_path);
  //warmup
  cv::Mat img(300, 300, CV_8UC3);
  detector.Detect(img);

  if (stat("./result", &st) != 0) {
      mkdir("./result", 0755);
  }

  std::cout<< "\n#######################################\n" <<std::endl;
  for (const auto& img_path: img_paths){
    std::cout<< img_path <<std::endl;

    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "[ERROR] Failed to read image: " << img_path << "\n";
      continue;
    }

    auto start = std::chrono::steady_clock::now();

    // Resize to model input size
    cv::resize(img, img, cv::Size(input_shape, input_shape), 0, 0, cv::INTER_CUBIC);

    auto objects = detector.Detect(img);
    detector.visualize_result(objects, img);

    auto end = std::chrono::steady_clock::now();

    std::cout<< "total time spent(ms): "
             << std::chrono::duration<float, std::milli>(end - start).count()
             <<std::endl;

    std::string img_name = img_path.substr(img_path.find_last_of("/\\") + 1);
    std::string result_name = "./result/" + img_name.substr(0, img_name.find_last_of('.')) + "_result.jpg";
    cv::imwrite(result_name, img);
    std::cout << "Result saved to " << result_name << std::endl;
    std::cout<< "\n#######################################\n" <<std::endl;
  }
}

void detect_camera_frame(Detector& detector){
  Server svr;
  svr.new_task_queue = [] { return new ThreadPool(1); };
  svr.Post("/predict", [&detector](const Request &req, Response &res) {
    auto image_file = req.get_file_value("image_file");
    // Pi sends raw 300×300 grayscale pixels (no JPEG encode/decode, no cvtColor)
    if (image_file.content.size() != 300 * 300) {
      std::cerr << "[ERROR] Invalid image data size: " << image_file.content.size() << "\n";
      res.set_content("{\"len\":0,\"action\":\"OK\",\"result\":[]}", "application/json");
      return;
    }
    cv::Mat img_decode(300, 300, CV_8UC1, (void*)image_file.content.data());

    if (img_decode.empty()) {
      std::cerr << "[ERROR] Failed to decode image\n";
      res.set_content("{\"len\":0,\"result\":[]}", "application/json");
      return;
    }

    auto start = std::chrono::steady_clock::now();
    auto objects = detector.Detect(img_decode);
    auto end = std::chrono::steady_clock::now();
    float prediction_time = std::chrono::duration<float, std::milli>(end - start).count();

    // Build JSON response matching HaoYao GrabImage expectations
    std::ostringstream json;
    bool has_defect = objects.size() > 0;
    json << "{\"len\":" << objects.size()
         << ",\"action\":\"" << (has_defect ? "NG" : "OK") << "\""
         << ",\"result\":[";
    for (size_t i = 0; i < objects.size(); i++) {
      if (i > 0) json << ",";
      auto& obj = objects[i];
      json << "{"
           << "\"class_name\":\"" << detector.GetClassName(obj.class_id) << "\","
           << "\"loc\":[" << obj.rec.x << "," << obj.rec.y << ","
           << (obj.rec.x + obj.rec.width) << "," << (obj.rec.y + obj.rec.height) << "],"
           << "\"score\":" << obj.prob << ","
           << "\"prediction_time\":" << prediction_time
           << "}";
    }
    json << "]}";

    res.set_content(json.str(), "application/json");
  });
  svr.listen("0.0.0.0", 8080);
}

void int_handler(int sig){
  fflush(stdout);
  fpga_release();
  std::cerr << "SIGINT: stopping the predictor\n";
  exit(-1);
}

int main(int argc, char** argv) {
  if (argc < 2){
    std::cerr << "\n[ERROR] Missing required configuration path argument\n";
    std::cerr << "Usage: " << argv[0] << " <config_path> [image_path|image_dir]\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  <config_path>      Required: Path to configuration file\n";
    std::cerr << "  [image_path|image_dir] Optional: Single image file or directory path\n";
    return 1;
  }
  std::string config_path = argv[1];

  signal(SIGINT, int_handler);

  Detector detector(config_path);

  if (argc == 2) detect_camera_frame(detector);
  else{
    std::string input_path = argv[2];
    detect_image_file(detector, input_path);
  }

  return 0;
}
