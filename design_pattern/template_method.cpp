// 程序库开发人员
class Library {
public:
  // 稳定的template method
  void run() {
    step1();

    if (step2()) { // 支持变化 ==> 虚函数多态调用
      step3();
    }

    for (int i = 0; i < 4; i++) {
      step4(); // 支持变化 ==> 虚函数多态调用
    }

    step5();
  }

  virtual ~Library(){};

protected:
  void step1() {
    // 稳定
  }

  void step3() {
    // 稳定
  }

  void step5() {
    // 稳定
  }

  virtual bool step2() = 0;
  virtual void step4() = 0;
};

// 应用程序开发人员
class Application : public Library {
protected:
  virtual bool step2() override {
    // ...子类重写实现
    return true;
  }

  virtual void step4() override {
    // ...子类重写实现
  }
};

int main() {
  Library *pLib = new Application();
  pLib->run();
  delete pLib;
}