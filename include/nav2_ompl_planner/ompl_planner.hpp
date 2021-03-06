/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2020 Shivang Patel
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of Willow Garage, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Shivang Patel
 *********************************************************************/

#ifndef NAV2_OMPL_PLANNER__OMPL_PLANNER_HPP_
#define NAV2_OMPL_PLANNER__OMPL_PLANNER_HPP_

#include <string>
#include <memory>
#include <type_traits>

#include "ompl/base/State.h"
#include "ompl/base/StateSpace.h"
#include "ompl/geometric/SimpleSetup.h"
#include "ompl/geometric/planners/rrt/RRTstar.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "nav2_core/global_planner.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"

namespace nav2_ompl_planner
{

namespace ob = ompl::base;
namespace og = ompl::geometric;

class OMPLPlanner : public nav2_core::GlobalPlanner
{
public:
  OMPLPlanner();
  ~OMPLPlanner() = default;

  // plugin configure
  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  // plugin cleanup
  void cleanup() override;

  // plugin activate
  void activate() override;

  // plugin deactivate
  void deactivate() override;

  // This method creates path for given start and goal pose.
  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

  // Sets planner
  template<typename Planner>
  void setPlanner()
  {
    ss_->setPlanner(std::make_shared<Planner>(ss_->getSpaceInformation()));
  }

  // Waypoint conversion
  geometry_msgs::msg::PoseStamped convertWaypoints(const ompl::base::State & state);

  void initialize();

  // State validity checker
  bool isStateValid(const ompl::base::State * state);  // ss set stateValidityCheck

private:


  // The global frame of the costmap
  std::string global_frame_, name_;

  nav2_util::LifecycleNode::SharedPtr node_;
  rclcpp::Logger logger_;

  std::shared_ptr<tf2_ros::Buffer> tf_;

  // Global Costmap
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;

  // Setup Ptr
  og::SimpleSetupPtr ss_;

  // State Space Ptr
  ob::StateSpacePtr ompl_state_space_;

  double solve_time_;   // ompl termination condition
  double collision_checking_resolution_; // setup ss stateValid check

  std::string planner_name_;  // ompl inner algrithm alias

  std::shared_ptr<nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>
  collision_checker_;
  bool allow_unknown_;

  bool is_initialized_;
};

}  // namespace nav2_ompl_planner

#endif  // NAV2_OMPL_PLANNER__OMPL_PLANNER_HPP_
