/*
 *  swiftpro_rviz_node.cpp
 *  swiftpro_description
 *  
 *  Created by Patryk Cieslak on 05/02/2025.
 *  Copyright (c) 2025 Patryk Cieslak. All rights reserved.
 */

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#define MATH_PI 				3.141592653589793238463
#define MATH_TRANS  			57.2958    
#define MATH_L1 				106.6
#define MATH_L2 				13.2
#define MATH_LOWER_ARM 			142.07
#define MATH_UPPER_ARM 			158.81	
#define MATH_UPPER_LOWER 		(MATH_UPPER_ARM / MATH_LOWER_ARM)

#define LOWER_ARM_MAX_ANGLE     135.6
#define LOWER_ARM_MIN_ANGLE     0
#define UPPER_ARM_MAX_ANGLE     100.7
#define UPPER_ARM_MIN_ANGLE     0
#define LOWER_UPPER_MAX_ANGLE   151
#define LOWER_UPPER_MIN_ANGLE   10

using namespace std::placeholders;

class SwiftProRVIZNode : public rclcpp::Node
{
public:
	SwiftProRVIZNode() : Node("swiftpro_rviz")
	{
		passive_joint_angle_.fill(0.0);

		// Params
		this->declare_parameter("manipulator_namespace", "swiftpro");
		manipulator_namespace_ = this->get_parameter("manipulator_namespace").as_string();
		
		// Topics
		subscription_ = this->create_subscription<sensor_msgs::msg::JointState>("joint_states", 10, 
			std::bind(&SwiftProRVIZNode::jointStateCallback, this, _1));
		publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
	}

	void updatePassiveJoints(const std::array<float,3>& active_joint_angle)
	{
		double alpha2 = 90 - active_joint_angle[1];
		double alpha3 = active_joint_angle[2] - 3.8;
		passive_joint_angle_[0] = (alpha2 + alpha3) - 176.11 + 90;
		passive_joint_angle_[1] = -90 + alpha2;
		passive_joint_angle_[2] = active_joint_angle[1];
		passive_joint_angle_[3] = 90 - (alpha2 + alpha3 + 3.8);
		passive_joint_angle_[4] = 176.11 - 180 - alpha3;
		passive_joint_angle_[5] = 48.39 + alpha3 - 44.55;
	}

	void jointStateCallback(const sensor_msgs::msg::JointState& in)
	{
		if(in.position.size() == 4)
		{
			std::array<float, 3> active_joint_angle;
			active_joint_angle[0] = in.position[0]/MATH_PI*180.f;
			active_joint_angle[1] = in.position[1]/MATH_PI*180.f;
			active_joint_angle[2] = in.position[2]/MATH_PI*180.f;
			updatePassiveJoints(active_joint_angle);

			sensor_msgs::msg::JointState out;
			out.header.stamp = in.header.stamp;
			out.name.resize(6);
			out.position.resize(6);
			out.name[0] = manipulator_namespace_ + "/passive_joint1";
			out.position[0] = passive_joint_angle_[0] / 57.2958;
			out.name[1] = manipulator_namespace_ + "/passive_joint2";
			out.position[1] = passive_joint_angle_[1] / 57.2958;
			out.name[2] = manipulator_namespace_ + "/passive_joint3";
			out.position[2] = passive_joint_angle_[2] / 57.2958;
			out.name[3] = manipulator_namespace_ + "/passive_joint5";
			out.position[3] = passive_joint_angle_[3] / 57.2958;
			out.name[4] = manipulator_namespace_ + "/passive_joint7";
			out.position[4] = passive_joint_angle_[4] / 57.2958;
			out.name[5] = manipulator_namespace_ + "/passive_joint8";
			out.position[5] = passive_joint_angle_[5] / 57.2958;	
			publisher_->publish(out);	
		}
	}

private:
	std::array<float, 6> passive_joint_angle_;
	std::string manipulator_namespace_;
	rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
	rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{	
	rclcpp::init(argc, argv);
  	auto swiftpro_rviz_node = std::make_shared<SwiftProRVIZNode>();
  	rclcpp::spin(swiftpro_rviz_node);
 	rclcpp::shutdown();
	return 0;
}

