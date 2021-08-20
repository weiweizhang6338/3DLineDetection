#include <pcl/io/pcd_io.h>
#include <stdio.h>

#include <fstream>

#include "LineDetection3D.h"
#include "Timer.h"
#include "nanoflann.hpp"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

void readDataFromFile(string filepath, PointCloud<double> &cloud) {
  cout << "Reading data ..." << endl;

  pcl::PointCloud<pcl::PointXYZI>::Ptr origin_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile<pcl::PointXYZI>(filepath, *origin_cloud);
  cloud.pts.resize(origin_cloud->size());
  for (int i = 0; i < origin_cloud->size(); i++) {
    const pcl::PointXYZI &pt = origin_cloud->at(i);
    cloud.pts[i] = PointCloud<double>::PtData(pt.x, pt.y, pt.z);
  }
  cout << "Total num of points: " << cloud.pts.size() << endl;
}

void writeOutPlanes(string filePath, vector<PLANE> &planes, double scale) {
  // write out bounding polygon result
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  for (int p = 0; p < planes.size(); ++p) {
    int R = rand() % 255;
    int G = rand() % 255;
    int B = rand() % 255;

    for (int i = 0; i < planes[p].lines3d.size(); ++i) {
      for (int j = 0; j < planes[p].lines3d[i].size(); ++j) {
        cv::Point3d dev =
            planes[p].lines3d[i][j][1] - planes[p].lines3d[i][j][0];
        double L = sqrt(dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
        int k = L / (scale / 10);

        double x = planes[p].lines3d[i][j][0].x,
               y = planes[p].lines3d[i][j][0].y,
               z = planes[p].lines3d[i][j][0].z;
        double dx = dev.x / k, dy = dev.y / k, dz = dev.z / k;

        for (int j = 0; j < k; ++j) {
          x += dx;
          y += dy;
          z += dz;

          pcl::PointXYZRGB pt;
          pt.x = x;
          pt.y = y;
          pt.z = z;
          pt.r = R;
          pt.g = G;
          pt.b = B;

          planes_cloud->push_back(pt);
        }
      }
    }
  }

  string save_path = filePath + "planes.pcd";
  pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(save_path, *planes_cloud);
}

void writeOutLines(string filePath, vector<vector<cv::Point3d> > &lines,
                   double scale) {
  // write out bounding polygon result
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr lines_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  for (int p = 0; p < lines.size(); ++p) {
    int R = rand() % 255;
    int G = rand() % 255;
    int B = rand() % 255;

    cv::Point3d dev = lines[p][1] - lines[p][0];
    double L = sqrt(dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
    int k = L / (scale / 10);

    double x = lines[p][0].x, y = lines[p][0].y, z = lines[p][0].z;
    double dx = dev.x / k, dy = dev.y / k, dz = dev.z / k;
    for (int j = 0; j < k; ++j) {
      x += dx;
      y += dy;
      z += dz;

      pcl::PointXYZRGB pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;
      pt.r = R;
      pt.g = G;
      pt.b = B;

      lines_cloud->push_back(pt);
    }
  }

  string save_path = filePath + "lines.pcd";
  pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(save_path, *lines_cloud);
}

int main() {
  string fileData =
      "/media/zw/Dataset/final_data/calibration_data/livox_camera/fail_method/"
      "points.pcd";
  string fileOut =
      "/media/zw/Dataset/final_data/calibration_data/livox_camera/fail_method/";

  // read in data
  PointCloud<double> pointData;
  readDataFromFile(fileData, pointData);

  int k = 20;
  LineDetection3D detector;
  vector<PLANE> planes;
  vector<vector<cv::Point3d> > lines;
  vector<double> ts;
  detector.run(pointData, k, planes, lines, ts);
  cout << "lines number: " << lines.size() << endl;
  cout << "planes number: " << planes.size() << endl;

  // writeOutPlanes(fileOut, planes, detector.scale);
  writeOutLines(fileOut, lines, detector.scale);

  return 0;
}