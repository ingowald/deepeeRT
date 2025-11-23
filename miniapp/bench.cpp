#include "deepee/deepee.h"
#include "DGEF.h"

int main(int ac, char **av)
{
  std::string rayFileName;
  std::string modelFileName;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg == "-r")
      rayFileName = av[++i];
    else if (arg == "-m")
      modelFileName = av[++i];
    else
      throw std::runtime_error("usage: ./bench"
                               " -r <rays>"
                               " -m <model>"
                               );
  }

  PRINT(modelFileName);
  dgef::Model::SP model = dgef::Model::read(modelFileName);
}
