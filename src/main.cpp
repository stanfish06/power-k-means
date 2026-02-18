#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <iostream>

int main() {
  using namespace boost::lambda;
  typedef std::istream_iterator<int> in;

  std::cout << "Enter number:";
  std::for_each(in(std::cin), in(), std::cout << (_1 * 3) << " ");
}
