/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

// for portability of M_PI (Vis Studio, MinGW, etc.)
#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	weights.resize(num_particles,1.0);

	normal_distribution<double> distx(x,std[0]);
	normal_distribution<double> disty(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	for(int i = 0; i < num_particles; ++i) {
		Particle p = Particle();
		p.id = i;
		p.x = distx(gen);
		p.y = disty(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights[i] = 1.0;
		particles.push_back (p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;
	double x, y, theta, newx, newy, newtheta;

	for(int i = 0; i < num_particles; ++i) {
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;
		if (fabs(yaw_rate) > 0.001) {
			newx = x + (velocity/yaw_rate)*(sin(theta + yaw_rate*delta_t) - sin(theta));
			newy = y + (velocity/yaw_rate)*(cos(theta) - cos(theta + yaw_rate*delta_t));
		} else {
			newx = x + velocity*cos(theta)*delta_t;
			newy = y + velocity*sin(theta)*delta_t;
		}
		newtheta = theta + yaw_rate*delta_t;
		normal_distribution<double> distx(newx,std_pos[0]);
		normal_distribution<double> disty(newy,std_pos[1]);
		normal_distribution<double> dist_theta(newtheta,std_pos[2]);
		particles[i].x = distx(gen);
		particles[i].y = disty(gen);
		particles[i].theta = dist_theta(gen);
		//particles[i].weight = 1;
		//weights[i] = 1;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
	//double inf = numeric_limits<double>::infinity();
	double inf = 999999;

	for(int k = 0; k < observations.size(); ++k) {
		
		int closest_pred_index;
		double smallest_distance = inf;

		for (int i = 0; i < predicted.size(); i++) {
			double distance = dist(predicted[i].x, predicted[i].y, observations[k].x, observations[k].y);
			if(distance < smallest_distance) {
				observations[k].id = i;
				smallest_distance = distance; 
			}
		} 
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for(int i = 0; i < num_particles; ++i) {
		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;

		vector<LandmarkObs> observations_map;

		// Transform every observation from particle's to map's coordinates
		for(int j = 0; j < observations.size(); ++j) {
			int obs_id = observations[j].id;
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;

			// homogenous transformation
			double obs_x_map = px + obs_x*cos(ptheta) - obs_y*sin(ptheta);
			double obs_y_map = py + obs_y*cos(ptheta) + obs_x*sin(ptheta);

			LandmarkObs obs_map = {obs_id, obs_x_map, obs_y_map};

			observations_map.push_back(obs_map);
		}

		vector<LandmarkObs> landmarks_in_range;

		// Find the map landmarks in range of the sensor
		for(int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;

			double distance = dist(px, py, landmark_x, landmark_y);

			if(distance < sensor_range) {
				LandmarkObs landmark_in_range = {landmark_id, landmark_x, landmark_y};
				landmarks_in_range.push_back(landmark_in_range);
			}

		}


		// Find the closest map landmark in range for every observation and store it 
		// in the observation id 
		dataAssociation(landmarks_in_range, observations_map);

		// Compare the observations and the landmarks in range to update the 
		// particle weight, i.e. the probability of being in the actual car's location

		double w = 1;

		for(int j = 0; j < observations_map.size(); ++j) {

			int obs_id_n = observations_map[j].id;
			double obs_x_n = observations_map[j].x;
			double obs_y_n = observations_map[j].y;

			//cout << landmarks_in_range.size() << endl;
			//cout << obs_id_n << endl;

			double landmark_x_n = landmarks_in_range[obs_id_n].x;
			double landmark_y_n = landmarks_in_range[obs_id_n].y;

			double prob = multivariate_norm(obs_x_n, landmark_x_n, obs_y_n, landmark_y_n, std_landmark[0], std_landmark[1]);

			w *= prob;
		} 

		particles[i].weight = w;
		weights[i] = w;
	}
}

void ParticleFilter::resample() {

	default_random_engine gen;

	discrete_distribution<int> d(weights.begin(), weights.end());
	vector<Particle> new_particles;
	for(int p=0; p<num_particles; ++p) {
		int new_particle_index = d(gen);
		new_particles.push_back (particles[new_particle_index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
