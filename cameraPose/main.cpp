#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<fstream>
#include<math.h>
using namespace std;
double sumVector(vector<double> x)
{
     double sum=0.0;
     for(int i=0; i<x.size();++i)
     {
         sum+=x[i];
     }
     return sum/x.size();
}
// 代价函数的计算模型
struct Resection
{
    Resection ( double X,double Y,double Z,double x, double y,double f ) :_X(X),_Y(Y),_Z(Z), _x ( x ), _y ( y ),_f(f) {}
    // 残差的计算
    template <typename T>
    bool operator() (const T* const camPose, T* residual ) const     // 残差
    {
            T Xs=camPose[0];
            T Ys=camPose[1];
            T Zs=camPose[2];
            T w=camPose[3];
            T p=camPose[4];
            T k=camPose[5];
            T a1=cos(k)*cos(p);
            T b1=-sin(k)*cos(w) + sin(p)*sin(w)*cos(k);
            T c1=sin(k)*sin(w) + sin(p)*cos(k)*cos(w);
            T a2=sin(k)*cos(p);
            T b2=sin(k)*sin(p)*sin(w) + cos(k)*cos(w);
            T c2=sin(k)*sin(p)*cos(w) - sin(w)*cos(k);
            T a3=-sin(p);
            T b3=sin(w)*cos(p);
            T c3=cos(p)*cos(w);
//            R=rotationVectorToMatrix(omega,pho,kappa);
            residual[0]=T(_x)-T(_f)*T((a1*(_X-Xs)+b1*(_Y-Ys)+c1*(_Z-Zs))/(a3*(_X-Xs)+b3*(_Y-Ys)+c3*(_Z-Zs)));
            residual[1]=T(_y)-T(_f)*T((a2*(_X-Xs)+b2*(_Y-Ys)+c2*(_Z-Zs))/(a3*(_X-Xs)+b3*(_Y-Ys)+c3*(_Z-Zs)));
            return true; //千万不要写成return 0,要写成return true
    }
private:
    const double _x;
    const double _y;
    const double _f;
    const double _X;
    const double _Y;
    const double _Z;
};

int main ( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);
    //read file
    string filename=argv[1];
    ifstream fin(filename.c_str());
    string line;
    vector<double> x;
    vector<double> y;
    vector<double> X;
    vector<double> Y;
    vector<double> Z;
    while(getline(fin,line))
    {
        char* pEnd;
        double a,b,c,d,e;
        a=strtod(line.c_str(),&pEnd);
        b=strtod(pEnd,&pEnd);
        c=strtod(pEnd,&pEnd);
        d=strtod(pEnd,&pEnd);
        e=strtod(pEnd,nullptr);
        x.push_back(a);
        y.push_back(b);
        X.push_back(c);
        Y.push_back(d);
        Z.push_back(e);
    }
    //初始化参数
    double camPose[6]={0};
    camPose[0]=sumVector(X);
    camPose[1]=sumVector(Y);
    camPose[2]=sumVector(Z);
    double f = 153.24; //mm
//    camPose[2]=50*f;
    //构建最小二乘
    ceres::Problem problem;
    try
    {
        for(int i=0;i<x.size();++i)
        {
            ceres::CostFunction *costfunction=new ceres::AutoDiffCostFunction<Resection,2,6>(new Resection(X[i],Y[i],Z[i],x[i]/1000,y[i]/1000,f/1000));
            //将残差方程和观测值加入到problem,nullptr表示核函数为无，
            problem.AddResidualBlock(costfunction,nullptr,camPose);
        }
    }
    catch(...)
    {
        cout<<"costFunction error"<<endl;
    }

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout
//    options.max_num_iterations=25;
    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"*************结果****************"<<endl;
    cout<<"estimated Xs,Ys,Zs,omega,pho,kappa = ";
    for ( auto p:camPose) cout<<p<<" ";
    cout<<endl;

    return 0;
}

