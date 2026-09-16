#include "ParameterParser.h"

namespace ti_mmwave_rospkg {

PLUGINLIB_EXPORT_CLASS(ti_mmwave_rospkg::ParameterParser, nodelet::Nodelet);

ParameterParser::ParameterParser() {}

void ParameterParser::onInit() {}

void ParameterParser::ParamsParser(ti_mmwave_rospkg::mmWaveCLI &srv, ros::NodeHandle &nh) {

    //   ROS_ERROR("%s",srv.request.comm.c_str());
    //   ROS_ERROR("%s",srv.response.resp.c_str());
    std::vector <std::string> v;
    std::string s = srv.request.comm.c_str(); 
    std::istringstream ss(s);
    std::string token;
    std::string req;
    int i = 0;
    while (std::getline(ss, token, ' ')) {
        v.push_back(token);
        if (i == 0) {
            // First token is the command name
            req = token;
        } else {
            if (!req.compare("profileCfg")) {
                switch (i) {
                    case 2:
                        nh.setParam("/ti_mmwave/startFreq", std::stof(token)); break;
                    case 3:
                        nh.setParam("/ti_mmwave/idleTime", std::stof(token)); break;
                    case 4:
                        nh.setParam("/ti_mmwave/adcStartTime", std::stof(token)); break;
                    case 5:
                        nh.setParam("/ti_mmwave/rampEndTime", std::stof(token)); break;
                    case 8:
                        nh.setParam("/ti_mmwave/freqSlopeConst", std::stof(token)); break;
                    case 10:
                        nh.setParam("/ti_mmwave/numAdcSamples", std::stoi(token)); break;
                    case 11:
                        nh.setParam("/ti_mmwave/digOutSampleRate", std::stof(token)); break;
                    case 14:
                        nh.setParam("/ti_mmwave/rxGain", std::stof(token)); break;
                }
            } else if (!req.compare("frameCfg")) {
                switch (i) {
                case 1:
                    nh.setParam("/ti_mmwave/chirpStartIdx", std::stoi(token)); break;
                case 2:
                    nh.setParam("/ti_mmwave/chirpEndIdx", std::stoi(token)); break;
                case 3:
                    nh.setParam("/ti_mmwave/numLoops", std::stoi(token)); break;
                case 4:
                    nh.setParam("/ti_mmwave/numFrames", std::stoi(token)); break;
                case 5:
                    nh.setParam("/ti_mmwave/framePeriodicity", std::stof(token)); break;
                }
            } else if (!req.compare("zoneDef")) {

	      switch (i) {
		case 2:
                    nh.setParam("/ti_mmwave/zoneMinX", std::stoi(token)); break;
                case 3:
                    nh.setParam("/ti_mmwave/zoneMaxX", std::stoi(token)); break;
		case 4:
                    nh.setParam("/ti_mmwave/zoneMinY", std::stoi(token)); break;
                case 5:
		    nh.setParam("/ti_mmwave/zoneMaxY", std::stoi(token)); break;
		case 6:
                    nh.setParam("/ti_mmwave/zoneMinZ", std::stoi(token)); break;
                case 7:
		    nh.setParam("/ti_mmwave/zoneMaxZ", std::stoi(token)); break;		    

	      }
	    }
        }
        i++;
    }
}

void ParameterParser::CalParams(ros::NodeHandle &nh) {
    float c0 = 299792458;
    int chirpStartIdx;
    int chirpEndIdx;
    int numLoops;
    float framePeriodicity;
    float startFreq;
    float idleTime;
    float adcStartTime;
    float rampEndTime;
    float digOutSampleRate;
    float freqSlopeConst;
    float numAdcSamples;
    float zoneMinX;
    float zoneMaxX;
    float zoneMinY;
    float zoneMaxY;
    float zoneMinZ;
    float zoneMaxZ;

    // Get raw parameters and check if they exist
    bool hasAllParams = true;
    hasAllParams &= nh.getParam("/ti_mmwave/startFreq", startFreq);
    hasAllParams &= nh.getParam("/ti_mmwave/idleTime", idleTime);
    hasAllParams &= nh.getParam("/ti_mmwave/adcStartTime", adcStartTime);
    hasAllParams &= nh.getParam("/ti_mmwave/rampEndTime", rampEndTime);
    hasAllParams &= nh.getParam("/ti_mmwave/digOutSampleRate", digOutSampleRate);
    hasAllParams &= nh.getParam("/ti_mmwave/freqSlopeConst", freqSlopeConst);
    hasAllParams &= nh.getParam("/ti_mmwave/numAdcSamples", numAdcSamples);

    hasAllParams &= nh.getParam("/ti_mmwave/chirpStartIdx", chirpStartIdx);
    hasAllParams &= nh.getParam("/ti_mmwave/chirpEndIdx", chirpEndIdx);
    hasAllParams &= nh.getParam("/ti_mmwave/numLoops", numLoops);
    hasAllParams &= nh.getParam("/ti_mmwave/framePeriodicity", framePeriodicity);

    if (!hasAllParams) {
        ROS_ERROR("ParameterParser::CalParams: Failed to get all required raw parameters!");
        ROS_ERROR("Make sure profileCfg and frameCfg commands were processed successfully.");
    }

    ROS_INFO("Raw parameters: startFreq=%.2f, idleTime=%.2f, rampEndTime=%.2f, digOutSampleRate=%.2f, freqSlopeConst=%.2f",
             startFreq, idleTime, rampEndTime, digOutSampleRate, freqSlopeConst);
    ROS_INFO("Raw parameters: numAdcSamples=%.0f, chirpStartIdx=%d, chirpEndIdx=%d, numLoops=%d, framePeriodicity=%.3f",
             numAdcSamples, chirpStartIdx, chirpEndIdx, numLoops, framePeriodicity);

    int ntx = chirpEndIdx - chirpStartIdx + 1;
    int nd = numLoops;
    int nr = numAdcSamples;
    float tfr = framePeriodicity * 1e-3;
    float fs = digOutSampleRate * 1e3;
    float kf = freqSlopeConst * 1e12;
    float adc_duration = nr / fs;
    float BW = adc_duration * kf;
    float PRI = (idleTime + rampEndTime) * 1e-6;
    float fc = startFreq * 1e9 + kf * (adcStartTime * 1e-6 + adc_duration / 2); 

    float vrange = c0 / (2 * BW);
    float max_range = nr * vrange;
    float max_vel = c0 / (2 * fc * PRI) / ntx;
    float vvel = max_vel / nd;

    ROS_INFO("Calculated parameters: ntx=%d, fs=%.2f, fc=%.2f, BW=%.2f, PRI=%.6f",
             ntx, fs, fc, BW, PRI);
    ROS_INFO("Calculated parameters: vrange=%.4f, max_range=%.2f, max_vel=%.2f, vvel=%.4f",
             vrange, max_range, max_vel, vvel);

    nh.setParam("/ti_mmwave/num_TX", ntx);
    nh.setParam("/ti_mmwave/f_s", fs);
    nh.setParam("/ti_mmwave/f_c", fc);
    nh.setParam("/ti_mmwave/BW", BW);
    nh.setParam("/ti_mmwave/PRI", PRI);
    nh.setParam("/ti_mmwave/t_fr", tfr);
    nh.setParam("/ti_mmwave/max_range", max_range);
    nh.setParam("/ti_mmwave/range_resolution", vrange);
    nh.setParam("/ti_mmwave/max_doppler_vel", max_vel);
    nh.setParam("/ti_mmwave/doppler_vel_resolution", vvel);

}
}
