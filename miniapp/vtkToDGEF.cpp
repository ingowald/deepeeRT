
/* iw - quick tool to read at least some types of vtu files (currently
   supportin only all-hex and only per-cell data files */

#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>

#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkNamedColors.h>
#include "DGEF.h"

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif

using namespace cuBQL;

dgef::Mesh::SP readFile(const std::string fileName)
{  
  std::vector<double> vertex;
  std::vector<double> perCellValue;
  std::vector<size_t> hex_index;

  dgef::Mesh::SP mesh = std::make_shared<dgef::Mesh>();
  
  std::cout << "parsing vtu file " << fileName << std::endl;
  //read all the data from the file
  vtkSmartPointer<vtkUnstructuredGridReader> reader =
    vtkSmartPointer<vtkUnstructuredGridReader>::New();
  reader->SetFileName(fileName.c_str());
  reader->Update();
 
  vtkUnstructuredGrid *grid = reader->GetOutput();
  
  vtkPointData* pointData = grid->GetPointData();
  size_t firstIndexThisVTU = vertex.size() / 3;
  
  // ==================================================================
  const int numPoints = grid->GetNumberOfPoints();
  std::cout << " - found " << numPoints << " points" << std::endl;
  for (int pointID=0;pointID<numPoints;pointID++) {
    double point[3];
    grid->GetPoint(pointID,point);
    mesh->vertices.push_back({point[0],point[1],point[2]});
  }
  
  // ==================================================================
  const int numCells = grid->GetNumberOfCells();
  std::cout << " - found " << numCells << " cells" << std::endl;
  for (int cellID=0;cellID<numCells;cellID++) {
    vtkIdType cellPoints;
    const vtkIdType *pointIDs;
    grid->GetCellPoints(cellID,cellPoints,pointIDs);
    if (cellPoints != 3)
      continue;
    
    mesh->indices.push_back({
        (uint64_t)pointIDs[0],
        (uint64_t)pointIDs[1],
        (uint64_t)pointIDs[2]
      });
  }

  vtkCellData* cellData = grid->GetCellData();
  if (!cellData)
    throw std::runtime_error("could not read cell data ....");
  
  vtkDataArray *dataArray = cellData->GetArray(0);
  if (!dataArray)
    throw std::runtime_error("could not read data array from cell data");
  for (int i=0;i<numCells;i++)
    perCellValue.push_back(dataArray->GetTuple1(i));
  
  // std::cout << "-------------------------------------------------------" << std::endl;
  // std::cout << "done reading " << fileName << " : "
  //           << std::endl << "  " << (vertex.size()/3) << " vertices "
  //           << std::endl << "  " << (hex_index.size()/8) << " hexes" << std::endl;
  
  return mesh;
}


int main ( int argc, char *argv[] )
{
  std::string outFileName = "";
  std::vector<std::string> inFileNames;
  
  for (int i=1;i<argc;i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      outFileName = argv[++i];
    else
      inFileNames.push_back(arg);
  }
  
  if(inFileNames.empty() || outFileName.empty())
  {
    std::cerr << "Usage: " << argv[0] << " -o outfile <infiles.vtu>+" << std::endl;
    return EXIT_FAILURE;
  }

  dgef::Model::SP model = std::make_shared<dgef::Model>();
  for (auto fn : inFileNames) {
    dgef::Instance inst;
    inst.meshID = model->meshes.size();
    model->meshes.push_back(readFile(fn));
    model->instances.push_back(inst);
  }
  std::cout << "=======================================================" << std::endl;
  std::cout << "writing out result ..." << std::endl;
  std::cout << "=======================================================" << std::endl;
  model->write(outFileName);
  return 0;

}
