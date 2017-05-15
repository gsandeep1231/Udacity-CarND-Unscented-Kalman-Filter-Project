#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.2;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.02;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  
  n_x_ = 5;
  
  n_aug_ = 7;
  
  lambda_ = 3 - n_aug_;
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);
  
  weights_ = VectorXd(2*n_aug_+1);
  weights_[0] = lambda_/(lambda_ + n_aug_);
  for (int i=1; i< (2*n_aug_+1); i++) {
      weights_[i] = 1/(2*(lambda_ + n_aug_));
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
	  cout << "Sensor type: " << meas_package.sensor_type_ << endl;
	  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		  float cart_x;
		  float cart_y;
		  cart_x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
		  cart_y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
		  x_ << cart_x, cart_y, 0, 0, 0;
	  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		  x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
	  }
	  time_us_ = meas_package.timestamp_;
	  P_ = MatrixXd::Identity(n_x_, n_x_);
	  is_initialized_ = true;
	  cout << "Initial X:" << x_ << endl;
	  cout << "Initial P:" << P_ << endl;
	  return;
  }
  
  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;
  
  // Predict
  Prediction(delta_t);
  //cout << "Predicted X:\n" << x_ << endl;
  //cout << "Predicted P:\n" << P_ << endl;

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }
  //cout << "Updated X:\n" << x_ << endl;
  //cout << "Updated P:\n" << P_ << endl;
  //cout << "Done processing" << endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //cout << "Predicting for time interval: " << delta_t << endl;
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0, 0;
  //cout << "x_aug: " << x_aug << endl;
  
  //create augmented state covariance
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug.block(n_x_,n_x_,2,2) = Q;
  //cout << "P_aug: " << P_aug << endl;
  
  //create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i<n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + sqrt(3) * A.col(i);
      Xsig_aug.col(n_aug_+i+1) = x_aug - sqrt(3) * A.col(i);
  }
  //cout << "Xsig_aug:\n" << Xsig_aug << endl;
  
  //predict sigma points
  for (int i=0; i<2*n_aug_+1; i++) {
      float px = Xsig_aug(0,i);
      float py = Xsig_aug(1,i);
      float v  = Xsig_aug(2,i);
      float t  = Xsig_aug(3,i);
      float tr = Xsig_aug(4,i);
      float noise_acc = Xsig_aug(5,i);
      float noise_yaw = Xsig_aug(6,i);
      
      float px_pred, py_pred;
      if (tr != 0) {
          px_pred = px + (v/tr)*(sin(t+tr*delta_t) -sin(t)) + ((delta_t*delta_t)/2)*noise_acc*cos(t);
          py_pred = py + (v/tr)*(-cos(t+tr*delta_t) + cos(t)) + ((delta_t*delta_t)/2)*noise_acc*sin(t);
      } else {
          px_pred = px + v*delta_t*cos(t) + ((delta_t*delta_t)/2)*noise_acc*cos(t);
          py_pred = py + v*delta_t*sin(t) + ((delta_t*delta_t)/2)*noise_acc*sin(t);
      }
      Xsig_pred_(0,i) = px_pred;
      Xsig_pred_(1,i) = py_pred;
      Xsig_pred_(2,i) = v + delta_t*noise_acc;
      Xsig_pred_(3,i) = t + tr*delta_t + ((delta_t*delta_t)/2)*noise_yaw;
      Xsig_pred_(4,i) = tr + delta_t*noise_yaw;   
  }
  //cout << "Xsig_pred:\n" << Xsig_pred_ << endl;
  
  //predict mean x_
  for (int i=0; i<n_x_; i++) {
      float sum = 0;
      for (int j=0; j<(2*n_aug_+1); j++) {
          sum += weights_[j]*Xsig_pred_(i,j);
      }
      x_[i] = sum;
  }
  
  //predict covariance P_
  P_.fill(0.0);
  VectorXd Xtmp = VectorXd(n_x_);
  VectorXd diff = VectorXd(n_x_);
  for (int i=0; i< (2*n_aug_+1); i++) {
      Xtmp = Xsig_pred_.col(i);
      diff = Xtmp - x_;
      P_ += weights_[i] * (diff*diff.transpose()); 
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //cout << "Updating Lidar" << endl;
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  //transform sigma points into measurement space
  for (int i=0; i<2 * n_aug_ + 1; i++) {
      float px = Xsig_pred_(0,i);
      float py = Xsig_pred_(1,i);
      
      Zsig.col(i) << px, py;
  }
  
  //calculate mean predicted measurement
  for (int i=0; i<n_z; i++) {
      float sum = 0;
      for (int j=0; j<2 * n_aug_ + 1; j++) {
          sum += weights_(j)*Zsig(i,j);
      }
      z_pred(i) = sum;
  }
  
  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;
  
  VectorXd diff = VectorXd(n_z);
  for (int i=0; i<2 * n_aug_ + 1; i++) {
      diff = Zsig.col(i) - z_pred;
      S += weights_(i)*(diff*diff.transpose());
  }
  S += R;
  
  //create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;
  
  //cout << "Zsig:\n" << Zsig << endl;
  //cout << "z_pred:\n" << z_pred << endl;
  //cout << "z:\n" << z << endl;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();  
  }
  
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //residual
  VectorXd z_diff = z - z_pred;
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //cout << "Updating Radar" << endl;
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  //transform sigma points into measurement space
  for (int i=0; i<2 * n_aug_ + 1; i++) {
      float px = Xsig_pred_(0,i);
      float py = Xsig_pred_(1,i);
      float v  = Xsig_pred_(2,i);
      float t  = Xsig_pred_(3,i);
      float tr = Xsig_pred_(4,i);
      
      float rho = sqrt(px*px + py*py);
      float phi = atan2(py, px);
      float rho_dot;
      if (rho < 0.000001) {
          rho_dot = ((px*v*cos(t) + py*v*sin(t)))/0.001; 
      } else {
          rho_dot = ((px*v*cos(t) + py*v*sin(t)))/rho; 
      }
      Zsig.col(i) << rho, phi, rho_dot;
  }
  //cout << "Zsig:\n" << Zsig << endl;
  //calculate mean predicted measurement
  for (int i=0; i<n_z; i++) {
      float sum = 0;
      for (int j=0; j<2 * n_aug_ + 1; j++) {
          sum += weights_(j)*Zsig(i,j);
      }
      z_pred(i) = sum;
  }
  //cout << "z_pred:\n" << z_pred << endl;
  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;
  
  VectorXd diff = VectorXd(n_z);
  for (int i=0; i<2 * n_aug_ + 1; i++) {
      diff = Zsig.col(i) - z_pred;
      S += weights_(i)*(diff*diff.transpose());
  }
  S += R;
  //cout << "R:\n" << R << endl;
  //cout << "S:\n" << S << endl;
  
  //create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;
  //cout << "z:\n" << z << endl;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  //cout << "Tc:\n" << Tc << endl;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose(); 
  }
  
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //cout << "K:\n" << K << endl;
  //residual
  VectorXd z_diff = z - z_pred;
  //cout << "z_diff:\n" << z_diff << endl;
  
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  //cout << "Returning" << endl;
}
