#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>

template <typename, typename> struct type_to_idx {};

template <typename, std::size_t> struct idx_to_type {};

template <typename... Types> class my_variant {
public:
  template <typename T,
            typename = std::enable_if_t<(std::is_same_v<Types, T> || ...), int>>
  my_variant(T value) {
    idx = type_to_idx<my_variant<Types...>, T>::value;
    new (union_ptr.data()) T(value);
  }

  ~my_variant() { deconstructors[idx](union_ptr); }

  my_variant(const my_variant &other) {
    idx = other.idx;
    copy_constructors[idx](union_ptr, other.union_ptr);
  }

  my_variant &operator=(const my_variant &other) {
    if (this != std::addressof(other)) {
      if (idx != other.idx) {
        throw std::bad_variant_access();
      }

      copy_assignors[idx](union_ptr, other.union_ptr);
    }
    return *this;
  }

  my_variant(my_variant &&other) {
    idx = other.idx;
    move_constructors[idx](union_ptr, other.union_ptr);
  }

  my_variant &operator=(my_variant &&other) {
    if (this != std::addressof(other)) {
      if (idx != other.idx) {
        throw std::bad_variant_access();
      }

      move_assignors[idx](union_ptr, other.union_ptr);
    }
    return *this;
  }

  template <std::size_t I> idx_to_type<my_variant, I>::type &get() {
    if (I != idx) {
      throw std::bad_variant_access();
    }
    using type = idx_to_type<my_variant, I>::type;
    return *reinterpret_cast<type *>(union_ptr.data());
  }

  template <typename T> T &get() {
    return get<type_to_idx<my_variant, T>::value>();
  }

  template <typename Lambda>
  std::common_type_t<std::invoke_result_t<Lambda, Types>...>
  visit(Lambda &&lambda) {
    visitors<Lambda>[idx](union_ptr, std::forward<Lambda>(lambda));
  }

  std::size_t index() { return idx; }

private:
  using union_type = std::array<char, std::max({sizeof(Types)...})>;

  using deconstruct_func = void (*)(union_type &);
  static inline deconstruct_func deconstructors[sizeof...(Types)] = {
      [](union_type &x) {
        Types *p = reinterpret_cast<Types *>(x.data());
        p->~Types();
      }...};

  using copy_construct_func = void (*)(union_type &, const union_type &);
  static inline copy_construct_func copy_constructors[sizeof...(Types)] = {
      [](union_type &dst, const union_type &src) {
        Types *dst_ptr = reinterpret_cast<Types *>(dst.data());
        const Types *src_ptr = reinterpret_cast<const Types *>(src.data());
        new (dst_ptr) Types(*src_ptr);
      }...};

  using copy_assign_func = void (*)(union_type &, const union_type &);
  static inline copy_assign_func copy_assignors[sizeof...(Types)] = {
      [](union_type &dst, const union_type &src) {
        Types *dst_ptr = reinterpret_cast<Types *>(dst.data());
        const Types *src_ptr = reinterpret_cast<const Types *>(src.data());
        *dst_ptr = *src_ptr;
      }...};

  using move_construct_func = void (*)(union_type &, union_type &);
  static inline move_construct_func move_constructors[sizeof...(Types)] = {
      [](union_type &dst, union_type &src) {
        Types *dst_ptr = reinterpret_cast<Types *>(dst.data());
        Types *src_ptr = reinterpret_cast<Types *>(src.data());
        new (dst_ptr) Types(std::move(*src_ptr));
      }...};

  using move_assign_func = void (*)(union_type &, union_type &);
  static inline move_assign_func move_assignors[sizeof...(Types)] = {
      [](union_type &dst, union_type &src) {
        Types *dst_ptr = reinterpret_cast<Types *>(dst.data());
        Types *src_ptr = reinterpret_cast<Types *>(src.data());
        *dst_ptr = std::move(*src_ptr);
      }...};

  template <typename Lambda>
  using visit_func =
      std::common_type_t<std::invoke_result_t<Lambda, Types>...> (*)(
          union_type &, Lambda &&);
  template <typename Lambda>
  static inline visit_func<Lambda> visitors[sizeof...(Types)] = {
      [](union_type &x, Lambda &&lambda) {
        Types *p = reinterpret_cast<Types *>(x.data());
        return std::invoke(std::forward<Lambda>(lambda), *p);
      }...};

  std::size_t idx;
  union_type union_ptr;
};

template <typename T, typename... Types>
struct type_to_idx<my_variant<T, Types...>, T> {
  static constexpr std::size_t value = 0;
};

template <typename T0, typename T1, typename... Types>
struct type_to_idx<my_variant<T0, Types...>, T1> {
  static constexpr std::size_t value =
      type_to_idx<my_variant<Types...>, T1>::value + 1;
};

template <typename T, typename... Types>
struct idx_to_type<my_variant<T, Types...>, 0> {
  using type = T;
};

template <typename T, typename... Types, std::size_t I>
struct idx_to_type<my_variant<T, Types...>, I> {
  using type = idx_to_type<my_variant<Types...>, I - 1>::type;
};

int main() {
  my_variant<int, std::string, double> object(std::string(
      "sdlskjljdskfjasdlkfjlaksdjflkasdjflkajsdflkjadslfkjadslkfj"));
  std::cout << "大小：" << sizeof(object) << std::endl;
  std::cout << "索引：" << object.index() << std::endl;
  std::cout << "数据by类型：" << object.get<std::string>() << std::endl;
  std::cout << "数据by索引：" << object.get<1>() << std::endl;

  std::cout << "##########################################" << std::endl;

  auto copy_object = object;
  std::cout << "拷贝对象，大小：" << sizeof(copy_object) << std::endl;
  std::cout << "拷贝对象，索引：" << copy_object.index() << std::endl;
  std::cout << "拷贝对象，数据by类型：" << copy_object.get<std::string>()
            << std::endl;
  std::cout << "拷贝对象，数据by索引：" << copy_object.get<1>() << std::endl;

  std::cout << "##########################################" << std::endl;

  auto move_object = std::move(object);
  std::cout << "移动对象，大小：" << sizeof(move_object) << std::endl;
  std::cout << "移动对象，索引：" << move_object.index() << std::endl;
  std::cout << "移动对象，数据by类型：" << move_object.get<std::string>()
            << std::endl;
  std::cout << "移动对象，数据by索引：" << move_object.get<1>() << std::endl;

  std::cout << "大小：" << sizeof(object) << std::endl;
  std::cout << "索引：" << object.index() << std::endl;
  std::cout << "数据by类型：" << object.get<std::string>() << std::endl;
  std::cout << "数据by索引：" << object.get<1>() << std::endl;

  std::cout << "##########################################" << std::endl;

  move_object.visit([](auto &&x) { std::cout << x << std::endl; });

  return 0;
}