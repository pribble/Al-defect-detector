#pragma once
#ifdef LITE_USE_THREAD_POOL
#include <atomic>
#include <condition_variable>  //NOLINT
#include <functional>
#include <mutex>   //NOLINT
#include <thread>  //NOLINT
#include <tuple>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {

class ThreadPool {
 public:
  typedef std::function<void(int, int)> TASK;
  typedef std::pair<std::function<void(int, int)>, int> TASK_BASIC;
  typedef std::tuple<std::function<void(int, int)>, int, int, int> TASK_COMMON;

  static void Enqueue(TASK_BASIC&& task);
  static void Enqueue(TASK_COMMON&& task);
  static void AcquireThreadPool();
  static void ReleaseThreadPool();
  static int Init(int number);
  static void Destroy();

 private:
  static ThreadPool* gInstance;
  explicit ThreadPool(int number = 0);
  ~ThreadPool();

  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  bool ready_{true};
  std::pair<TASK, std::vector<std::atomic<bool>*>> tasks_;
  std::condition_variable cv_;
  std::mutex mutex_;

  int thread_num_ = 0;
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_USE_THREAD_POOL
