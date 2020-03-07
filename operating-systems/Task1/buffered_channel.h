#ifndef BUFFERED_CHANNEL_H_
#define BUFFERED_CHANNEL_H_

#include <utility>
#include <queue>
#include <atomic>
#include <thread>
#include <condition_variable>

template<class T>
class BufferedChannel {
 public:
  explicit BufferedChannel(int size_) {
    capacity = size_;
    is_closed = false;
  }

  void Send(T value) {

    if (is_closed) {
      throw std::runtime_error("sending when closed");
    }
    std::unique_lock<std::mutex> locker(lock_queue);
    send_lock.wait(locker, [this] {
      return (queue_.size() < capacity) || is_closed;
    });
    if (is_closed) {
      throw std::runtime_error("sending when closed");
    }

    queue_.push(std::move(value));
    rcv_lock.notify_one();

  }

  std::pair<T, bool> Recv() {

    std::unique_lock<std::mutex> locker(lock_queue);
    if (is_closed) {
      if (!queue_.empty()) {
        std::pair<T, bool> tmp = std::make_pair(queue_.front(), true);
        queue_.pop();
        return tmp;
      } else {
        return std::make_pair(T(), false);
      }
    }
    rcv_lock.wait(locker, [this] { return !queue_.empty() || is_closed; });
    if (!queue_.empty()) {
      std::pair<T, bool> tmp = std::make_pair(queue_.front(), true);
      queue_.pop();
      send_lock.notify_one();
      return tmp;
    } else {
      return std::make_pair(T(), false);
    }

  }

  void Close() {
    is_closed = true;
    rcv_lock.notify_all();
    send_lock.notify_all();
  }

 private:
  std::queue<T> queue_;
  std::mutex lock_queue;
  std::condition_variable rcv_lock;
  std::condition_variable send_lock;
  std::atomic<int> capacity;
  std::atomic<bool> is_closed;

};

#endif // BUFFERED_CHANNEL_H_