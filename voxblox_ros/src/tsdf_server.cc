#include "voxblox_ros/tsdf_server.h"

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/ros_params.h"
#include <chrono>

namespace voxblox {

TsdfServer::TsdfServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private)
    : TsdfServer(nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
                 getTsdfIntegratorConfigFromRosParam(nh_private),
                 getMeshIntegratorConfigFromRosParam(nh_private)) {}

TsdfServer::TsdfServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       const TsdfMap::Config& config,
                       const TsdfIntegratorBase::Config& integrator_config,
                       const MeshIntegratorConfig& mesh_config)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(nh, nh_private) {
  getServerConfigFromRosParam(nh_private);

  // Advertise topics.
  surface_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
          "surface_pointcloud", 1, true);
  tsdf_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >("tsdf_pointcloud",
                                                              1, true);
  occupancy_marker_pub_ =
      nh_private_.advertise<visualization_msgs::MarkerArray>("occupied_nodes",
                                                             1, true);
  tsdf_slice_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
      "tsdf_slice", 1, true);

  nh_private_.param("pointcloud_queue_size", pointcloud_queue_size_,
                    pointcloud_queue_size_);
  pointcloud_sub_ = nh_.subscribe("pointcloud", pointcloud_queue_size_,
                                  &TsdfServer::insertPointcloud, this);

  mesh_pub_ = nh_private_.advertise<voxblox_msgs::Mesh>("mesh", 1, true);

  // Publishing/subscribing to a layer from another node (when using this as
  // a library, for example within a planner).
  tsdf_map_pub_ =
      nh_private_.advertise<voxblox_msgs::Layer>("tsdf_map_out", 1, false);
  tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 1,
                                        &TsdfServer::tsdfMapCallback, this);
  nh_private_.param("publish_tsdf_map", publish_tsdf_map_, publish_tsdf_map_);

  if (use_freespace_pointcloud_) {
    // points that are not inside an object, but may also not be on a surface.
    // These will only be used to mark freespace beyond the truncation distance.
    freespace_pointcloud_sub_ =
        nh_.subscribe("freespace_pointcloud", pointcloud_queue_size_,
                      &TsdfServer::insertFreespacePointcloud, this);
  }

  if (enable_icp_) {
    icp_transform_pub_ = nh_private_.advertise<geometry_msgs::TransformStamped>(
        "icp_transform", 1, true);
    nh_private_.param("icp_corrected_frame", icp_corrected_frame_,
                      icp_corrected_frame_);
    nh_private_.param("pose_corrected_frame", pose_corrected_frame_,
                      pose_corrected_frame_);
  }

  // Initialize TSDF Map and integrator.
  tsdf_map_.reset(new TsdfMap(config));

  std::string method("merged");
  nh_private_.param("method", method, method);
  if (method.compare("simple") == 0) {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("merged") == 0) {
    tsdf_integrator_.reset(new MergedTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("fast") == 0) {
    tsdf_integrator_.reset(new FastTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  }

  mesh_layer_.reset(new MeshLayer(tsdf_map_->block_size()));

  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()));

  icp_.reset(new ICP(getICPConfigFromRosParam(nh_private)));

  // compute frustum end points
  camera_param.max_range = 5.0;
  camera_param.min_range = 0.1;
  camera_param.fov << 90.0 * M_PI / 180.0, 65.0 * M_PI / 180.0;
  camera_param.resolution << 5.0 * M_PI / 180.0, 5.0 * M_PI / 180.0;
  camera_param.initialize();

  // param for decay function
  nh_private.param("decay_lambda", decay_lambda_, decay_lambda_);
  std::cout << "decay_lambda:" << decay_lambda_ << std::endl;
  nh_private.param("decay_distance", decay_distance_, decay_distance_);
  std::cout << "decay_distance:" << decay_distance_ << std::endl;

  sdf_layer_ = getTsdfMapPtr()->getTsdfLayerPtr();

  // Advertise services.
  calc_info_gain_srv_ = nh_private_.advertiseService(
      "calc_info_gain", &TsdfServer::calcInfoGainCallback, this);
  generate_mesh_srv_ = nh_private_.advertiseService(
      "generate_mesh", &TsdfServer::generateMeshCallback, this);
  clear_map_srv_ = nh_private_.advertiseService(
      "clear_map", &TsdfServer::clearMapCallback, this);
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &TsdfServer::saveMapCallback, this);
  load_map_srv_ = nh_private_.advertiseService(
      "load_map", &TsdfServer::loadMapCallback, this);
  publish_pointclouds_srv_ = nh_private_.advertiseService(
      "publish_pointclouds", &TsdfServer::publishPointcloudsCallback, this);
  publish_tsdf_map_srv_ = nh_private_.advertiseService(
      "publish_map", &TsdfServer::publishTsdfMapCallback, this);

  // If set, use a timer to progressively integrate the mesh.
  double update_mesh_every_n_sec = 1.0;
  nh_private_.param("update_mesh_every_n_sec", update_mesh_every_n_sec,
                    update_mesh_every_n_sec);

  if (update_mesh_every_n_sec > 0.0) {
    update_mesh_timer_ =
        nh_private_.createTimer(ros::Duration(update_mesh_every_n_sec),
                                &TsdfServer::updateMeshEvent, this);
  }

  double publish_map_every_n_sec = 1.0;
  nh_private_.param("publish_map_every_n_sec", publish_map_every_n_sec,
                    publish_map_every_n_sec);

  if (publish_map_every_n_sec > 0.0) {
    publish_map_timer_ =
        nh_private_.createTimer(ros::Duration(publish_map_every_n_sec),
                                &TsdfServer::publishMapEvent, this);
  }
}

void TsdfServer::getServerConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  // Before subscribing, determine minimum time between messages.
  // 0 by default.
  double min_time_between_msgs_sec = 0.0;
  nh_private.param("min_time_between_msgs_sec", min_time_between_msgs_sec,
                   min_time_between_msgs_sec);
  min_time_between_msgs_.fromSec(min_time_between_msgs_sec);

  nh_private.param("max_block_distance_from_body",
                   max_block_distance_from_body_,
                   max_block_distance_from_body_);
  nh_private.param("slice_level", slice_level_, slice_level_);
  nh_private.param("world_frame", world_frame_, world_frame_);
  nh_private.param("publish_pointclouds_on_update",
                   publish_pointclouds_on_update_,
                   publish_pointclouds_on_update_);
  nh_private.param("publish_slices", publish_slices_, publish_slices_);
  nh_private.param("publish_pointclouds", publish_pointclouds_,
                   publish_pointclouds_);

  nh_private.param("use_freespace_pointcloud", use_freespace_pointcloud_,
                   use_freespace_pointcloud_);
  nh_private.param("pointcloud_queue_size", pointcloud_queue_size_,
                   pointcloud_queue_size_);
  nh_private.param("enable_icp", enable_icp_, enable_icp_);
  nh_private.param("accumulate_icp_corrections", accumulate_icp_corrections_,
                   accumulate_icp_corrections_);

  nh_private.param("verbose", verbose_, verbose_);

  // Mesh settings.
  nh_private.param("mesh_filename", mesh_filename_, mesh_filename_);
  std::string color_mode("");
  nh_private.param("color_mode", color_mode, color_mode);
  color_mode_ = getColorModeFromString(color_mode);

  // Color map for intensity pointclouds.
  std::string intensity_colormap("rainbow");
  float intensity_max_value = kDefaultMaxIntensity;
  nh_private.param("intensity_colormap", intensity_colormap,
                   intensity_colormap);
  nh_private.param("intensity_max_value", intensity_max_value,
                   intensity_max_value);

  // Default set in constructor.
  if (intensity_colormap == "rainbow") {
    color_map_.reset(new RainbowColorMap());
  } else if (intensity_colormap == "inverse_rainbow") {
    color_map_.reset(new InverseRainbowColorMap());
  } else if (intensity_colormap == "grayscale") {
    color_map_.reset(new GrayscaleColorMap());
  } else if (intensity_colormap == "inverse_grayscale") {
    color_map_.reset(new InverseGrayscaleColorMap());
  } else if (intensity_colormap == "ironbow") {
    color_map_.reset(new IronbowColorMap());
  } else {
    ROS_ERROR_STREAM("Invalid color map: " << intensity_colormap);
  }
  color_map_->setMaxValue(intensity_max_value);
}

void TsdfServer::processPointCloudMessageAndInsert(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  // Convert the PCL pointcloud into our awesome format.

  // Horrible hack fix to fix color parsing colors in PCL.
  bool color_pointcloud = false;
  bool has_intensity = false;
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
      has_intensity = true;
    }
  }

  Pointcloud points_C;
  Colors colors;
  timing::Timer ptcloud_timer("ptcloud_preprocess");

  // Convert differently depending on RGB or I type.
  if (color_pointcloud) {
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else if (has_intensity) {
    pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else {
    pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  }
  ptcloud_timer.Stop();

  Transformation T_G_C_refined = T_G_C;
  if (enable_icp_) {
    timing::Timer icp_timer("icp");
    if (!accumulate_icp_corrections_) {
      icp_corrected_transform_.setIdentity();
    }
    static Transformation T_offset;
    const size_t num_icp_updates =
        icp_->runICP(tsdf_map_->getTsdfLayer(), points_C,
                     icp_corrected_transform_ * T_G_C, &T_G_C_refined);
    if (verbose_) {
      ROS_INFO("ICP refinement performed %zu successful update steps",
               num_icp_updates);
    }
    icp_corrected_transform_ = T_G_C_refined * T_G_C.inverse();

    if (!icp_->refiningRollPitch()) {
      // its already removed internally but small floating point errors can
      // build up if accumulating transforms
      Transformation::Vector6 T_vec = icp_corrected_transform_.log();
      T_vec[3] = 0.0;
      T_vec[4] = 0.0;
      icp_corrected_transform_ = Transformation::exp(T_vec);
    }

    // Publish transforms as both TF and message.
    tf::Transform icp_tf_msg, pose_tf_msg;
    geometry_msgs::TransformStamped transform_msg;

    tf::transformKindrToTF(icp_corrected_transform_.cast<double>(),
                           &icp_tf_msg);
    tf::transformKindrToTF(T_G_C.cast<double>(), &pose_tf_msg);
    tf::transformKindrToMsg(icp_corrected_transform_.cast<double>(),
                            &transform_msg.transform);
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(icp_tf_msg, pointcloud_msg->header.stamp,
                             world_frame_, icp_corrected_frame_));
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(pose_tf_msg, pointcloud_msg->header.stamp,
                             icp_corrected_frame_, pose_corrected_frame_));

    transform_msg.header.frame_id = world_frame_;
    transform_msg.child_frame_id = icp_corrected_frame_;
    icp_transform_pub_.publish(transform_msg);

    icp_timer.Stop();
  }

  if (verbose_) {
    ROS_INFO("Integrating a pointcloud with %lu points.", points_C.size());
  }

  ros::WallTime start = ros::WallTime::now();
  integratePointcloud(T_G_C_refined, points_C, colors, is_freespace_pointcloud);
  ros::WallTime end = ros::WallTime::now();
  if (verbose_) {
    ROS_INFO("Finished integrating in %f seconds, have %lu blocks.",
             (end - start).toSec(),
             tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  mesh_layer_->clearDistantMesh(T_G_C.getPosition(),
                                max_block_distance_from_body_);
  block_remove_timer.Stop();

  // Callback for inheriting classes.
  newPoseCallback(T_G_C);
}

// Checks if we can get the next message from queue.
bool TsdfServer::getNextPointcloudFromQueue(
    std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
    sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C) {
  const size_t kMaxQueueSize = 10;
  if (queue->empty()) {
    return false;
  }
  *pointcloud_msg = queue->front();
  if (transformer_.lookupTransform((*pointcloud_msg)->header.frame_id,
                                   world_frame_,
                                   (*pointcloud_msg)->header.stamp, T_G_C)) {
    queue->pop();
    return true;
  } else {
    if (queue->size() >= kMaxQueueSize) {
      ROS_ERROR_THROTTLE(60,
                         "Input pointcloud queue getting too long! Dropping "
                         "some pointclouds. Either unable to look up transform "
                         "timestamps or the processing is taking too long.");
      while (queue->size() >= kMaxQueueSize) {
        queue->pop();
      }
    }
  }
  return false;
}

void TsdfServer::insertPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  bool processed_any = false;
  while (
      getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = false;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
    processed_any = true;
  }

  if (!processed_any) {
    return;
  }

  if (publish_pointclouds_on_update_) {
    publishPointclouds();
  }

  if (verbose_) {
    ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    ROS_INFO_STREAM(
        "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
  }
}

void TsdfServer::insertFreespacePointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_freespace_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_freespace_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    freespace_pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  while (getNextPointcloudFromQueue(&freespace_pointcloud_queue_,
                                    &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = true;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
  }
}

void TsdfServer::integratePointcloud(const Transformation& T_G_C,
                                     const Pointcloud& ptcloud_C,
                                     const Colors& colors,
                                     const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  tsdf_integrator_->integratePointCloud(T_G_C, ptcloud_C, colors,
                                        is_freespace_pointcloud);
}

void TsdfServer::publishAllUpdatedTsdfVoxels() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  createDistancePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(), &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_pointcloud_pub_.publish(pointcloud);
}

void TsdfServer::publishTsdfSurfacePoints() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  const float surface_distance_thresh =
      tsdf_map_->getTsdfLayer().voxel_size() * 0.75;
  createSurfacePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(),
                                       surface_distance_thresh, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  surface_pointcloud_pub_.publish(pointcloud);
}

void TsdfServer::publishTsdfOccupiedNodes() {
  // Create a pointcloud with distance = intensity.
  visualization_msgs::MarkerArray marker_array;
  createOccupancyBlocksFromTsdfLayer(tsdf_map_->getTsdfLayer(), world_frame_,
                                     &marker_array);
  occupancy_marker_pub_.publish(marker_array);
}

void TsdfServer::publishSlices() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  createDistancePointcloudFromTsdfLayerSlice(tsdf_map_->getTsdfLayer(), 2,
                                             slice_level_, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_slice_pub_.publish(pointcloud);
}

void TsdfServer::publishMap(bool reset_remote_map) {
  if (!publish_tsdf_map_) {
    return;
  }
  int subscribers = this->tsdf_map_pub_.getNumSubscribers();
  if (subscribers > 0) {
    if (num_subscribers_tsdf_map_ < subscribers) {
      // Always reset the remote map and send all when a new subscriber
      // subscribes. A bit of overhead for other subscribers, but better than
      // inconsistent map states.
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_tsdf");
    voxblox_msgs::Layer layer_msg;
    serializeLayerAsMsg<TsdfVoxel>(this->tsdf_map_->getTsdfLayer(),
                                   only_updated, &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    this->tsdf_map_pub_.publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_tsdf_map_ = subscribers;
}

void TsdfServer::publishPointclouds() {
  // Combined function to publish all possible pointcloud messages -- surface
  // pointclouds, updated points, and occupied points.
  publishAllUpdatedTsdfVoxels();
  publishTsdfSurfacePoints();
  publishTsdfOccupiedNodes();
  if (publish_slices_) {
    publishSlices();
  }
}

void TsdfServer::updateMesh() {
  if (verbose_) {
    ROS_INFO("Updating mesh.");
  }

  timing::Timer generate_mesh_timer("mesh/update");
  constexpr bool only_mesh_updated_blocks = true;
  constexpr bool clear_updated_flag = true;
  mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");

  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  if (cache_mesh_) {
    cached_mesh_msg_ = mesh_msg;
  }

  publish_mesh_timer.Stop();

  if (publish_pointclouds_ && !publish_pointclouds_on_update_) {
    publishPointclouds();
  }
}

bool TsdfServer::generateMesh() {
  timing::Timer generate_mesh_timer("mesh/generate");
  const bool clear_mesh = true;
  if (clear_mesh) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  } else {
    constexpr bool only_mesh_updated_blocks = true;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  if (!mesh_filename_.empty()) {
    timing::Timer output_mesh_timer("mesh/output");
    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
    output_mesh_timer.Stop();
    if (success) {
      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output mesh as PLY: %s", mesh_filename_.c_str());
    }
  }

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool TsdfServer::saveMap(const std::string& file_path) {
  // Inheriting classes should add saving other layers to this function.
  return io::SaveLayer(tsdf_map_->getTsdfLayer(), file_path);
}

bool TsdfServer::loadMap(const std::string& file_path) {
  // Inheriting classes should add other layers to load, as this will only
  // load
  // the TSDF layer.
  constexpr bool kMulitpleLayerSupport = true;
  bool success = io::LoadBlocksFromFile(
      file_path, Layer<TsdfVoxel>::BlockMergingStrategy::kReplace,
      kMulitpleLayerSupport, tsdf_map_->getTsdfLayerPtr());
  if (success) {
    LOG(INFO) << "Successfully loaded TSDF layer.";
  }
  return success;
}

bool TsdfServer::clearMapCallback(std_srvs::Empty::Request& /*request*/,
                                  std_srvs::Empty::Response&
                                  /*response*/) {  // NOLINT
  clear();
  return true;
}

bool TsdfServer::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
                                      std_srvs::Empty::Response&
                                      /*response*/) {  // NOLINT
  return generateMesh();
}

bool TsdfServer::saveMapCallback(voxblox_msgs::FilePath::Request& request,
                                 voxblox_msgs::FilePath::Response&
                                 /*response*/) {  // NOLINT
  return saveMap(request.file_path);
}

bool TsdfServer::loadMapCallback(voxblox_msgs::FilePath::Request& request,
                                 voxblox_msgs::FilePath::Response&
                                 /*response*/) {  // NOLINT
  bool success = loadMap(request.file_path);
  return success;
}

bool TsdfServer::publishPointcloudsCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT
  publishPointclouds();
  return true;
}

bool TsdfServer::publishTsdfMapCallback(std_srvs::Empty::Request& /*request*/,
                                        std_srvs::Empty::Response&
                                        /*response*/) {  // NOLINT
  publishMap();
  return true;
}

void TsdfServer::updateMeshEvent(const ros::TimerEvent& /*event*/) {
  updateMesh();
}

void TsdfServer::publishMapEvent(const ros::TimerEvent& /*event*/) {
  publishMap();
}

void TsdfServer::clear() {
  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
  mesh_layer_->clear();
  interesting_voxels->clear();

  // Publish a message to reset the map to all subscribers.
  if (publish_tsdf_map_) {
    constexpr bool kResetRemoteMap = true;
    publishMap(kResetRemoteMap);
  }
}

void TsdfServer::tsdfMapCallback(const voxblox_msgs::Layer& layer_msg) {
  timing::Timer receive_map_timer("map/receive_tsdf");

  bool success =
      deserializeMsgToLayer<TsdfVoxel>(layer_msg, tsdf_map_->getTsdfLayerPtr());

  if (!success) {
    ROS_ERROR_THROTTLE(10, "Got an invalid TSDF map message!");
  } else {
    ROS_INFO_ONCE("Got an TSDF map from ROS topic!");
    if (publish_pointclouds_on_update_) {
      publishPointclouds();
    }
  }
}

void TsdfServer::integratePointcloud(const Transformation& T_G_C,
                                     const Pointcloud& ptcloud_C,
                                     const Colors& colors,
                                     const Interestingness& interestingness,
                                     const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  tsdf_integrator_->integratePointCloudWithInterestingness(T_G_C, ptcloud_C, colors,
                                        interestingness,
                                        is_freespace_pointcloud);                                      
}

void TsdfServer::lightProcessPointCloudMessageAndInsert(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg, std::shared_ptr<std::vector<GlobalIndex>> interesting_voxel_idx,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  Pointcloud points_C;
  Colors colors;
  Interestingness interestingness;
  timing::Timer ptcloud_timer("ptcloud_preprocess");
  pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
  // pointcloud_pcl is modified below:
  pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
  convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors, &interestingness);

  ptcloud_timer.Stop();

  Transformation T_G_C_refined = T_G_C;

  if (verbose_) {
    ROS_INFO("Integrating a pointcloud with %lu points.", points_C.size());
  }

  ros::WallTime start = ros::WallTime::now();
  integratePointcloud(T_G_C_refined, points_C, colors, interestingness, is_freespace_pointcloud);
  ros::WallTime end = ros::WallTime::now();

  // populate interesting voxel idx
  // filter points that have interestingness > threshold
  // std::cout << "PointCloud before filtering: " << pointcloud_pcl.width * pointcloud_pcl.height 
  //      << " data points (" << pcl::getFieldsList(pointcloud_pcl) << ")." << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_out(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_in(new pcl::PointCloud<pcl::PointXYZI>);
  *pcl_in = pointcloud_pcl;
  pcl::PassThrough<pcl::PointXYZI> pass_filter;
  pass_filter.setInputCloud(pcl_in);
  pass_filter.setFilterFieldName("intensity");
  pass_filter.setFilterLimits(0.5, 1.0);
  pass_filter.filter(*pcl_out);
  // std::cout << "PointCloud after pass filtering: " << pcl_out->width * pcl_out->height 
  //      << " data points (" << pcl::getFieldsList(*pcl_out) << ")." << std::endl;    
  
  // voxel filter
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setInputCloud(pcl_out);
  voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);
  voxel_filter.filter(*pcl_out);
  // std::cout << "PointCloud after voxel filtering: " << pcl_out->width * pcl_out->height 
  //      << " data points (" << pcl::getFieldsList(*pcl_out) << ")." << std::endl;  
  
  // get their global_voxel_idx
  const float voxel_size = sdf_layer_->voxel_size();
  for (auto p: pcl_out->points) {
    Point p_tmp;
    p_tmp << p.x, p.y, p.z;
    voxblox::GlobalIndex global_index = voxblox::getGridIndexFromPoint<voxblox::GlobalIndex>(
              p_tmp, 1.0/voxel_size);
    voxblox::TsdfVoxel* voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index);
    if (voxel != nullptr) {
      if (!voxel->in_queue) {
        voxel->in_queue = true;
        if (voxel->interestingness > 0.0) {
          voxel->interesting_distance = 0.0;
          // std::cout << "voxel->interestingness:" << voxel->interestingness;
          interesting_voxel_idx->push_back(global_index);
        }
      }
    }
  }

  if (verbose_) {
    ROS_INFO("Finished integrating in %f seconds, have %lu blocks.",
             (end - start).toSec(),
             tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks());
  }
}

void TsdfServer::lightInsertPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in,
    std::shared_ptr<std::vector<GlobalIndex>> interesting_voxel_idx) {
  Transformation T_G_C;
  T_G_C.setIdentity();

  constexpr bool is_freespace_pointcloud = false;
  lightProcessPointCloudMessageAndInsert(pointcloud_msg_in, interesting_voxel_idx, T_G_C,
                                    is_freespace_pointcloud);

  if (verbose_) {
    ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    ROS_INFO_STREAM(
        "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
  }
}

void SensorParamsBase::getFrustumEndpoints(StateVec& state,
                                           std::vector<Eigen::Vector3d>& ep) {
  // Convert rays from B to W.
  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(state[5], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(state[4], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(state[3], Eigen::Vector3d::UnitX());
  ep.clear();
  for (auto& p : frustum_endpoints_B) {
    Eigen::Vector3d p_tf = origin + rot_W2B * p;
    ep.push_back(p_tf);
  }
}

void SensorParamsBase::initialize() {
  double v_res, h_res;
  v_res = (resolution[1] > 0) ? resolution[1] : (1.0 * M_PI / 180.0);
  h_res = (resolution[0] > 0) ? resolution[0] : (1.0 * M_PI / 180.0);
  // Compute the normal vectors enclosed the FOV.

  // Compute 4 normal vectors for left, right, top, down planes.
  // First, compute 4 coner points.
  // Assume the range is along the hypotenuse of the right angle.
  double h_2 = fov[0] / 2;
  double v_2 = fov[1] / 2;
  Eigen::Vector3d pTL(cos(h_2), sin(h_2), sin(v_2));
  Eigen::Vector3d pTR(cos(h_2), -sin(h_2), sin(v_2));
  Eigen::Vector3d pBR(cos(h_2), -sin(h_2), -sin(v_2));
  Eigen::Vector3d pBL(cos(h_2), sin(h_2), -sin(v_2));
  edge_points.col(0) = pTL;
  edge_points.col(1) = pTR;
  edge_points.col(2) = pBR;
  edge_points.col(3) = pBL;
  // Compute normal vectors for 4 planes. (normalized)
  normal_vectors.col(0) = edge_points.col(0).cross(edge_points.col(1));
  normal_vectors.col(1) = edge_points.col(1).cross(edge_points.col(2));
  normal_vectors.col(2) = edge_points.col(2).cross(edge_points.col(3));
  normal_vectors.col(3) = edge_points.col(3).cross(edge_points.col(0));
  // Compute correct points based on the sensor range.
  edge_points = max_range * edge_points;
  // edge_points_B = rot_B2S * edge_points;
  edge_points_B = edge_points;
  // Frustum endpoints in (S) for gain calculation.
  frustum_endpoints.clear();
  frustum_endpoints_B.clear();
  int w = 0, h = 0;
  height = 0;
  width = 0;
  double h_lim_2 = fov[0] / 2;
  double v_lim_2 = fov[1] / 2;
  for (double dv = -v_lim_2; dv < v_lim_2; dv += v_res) {
    ++h;
    for (double dh = -h_lim_2; dh < h_lim_2; dh += h_res) {
      if (width == 0) {
        ++w;
      }
      double x = max_range * cos(dh);
      double y = max_range * sin(dh);
      double z = max_range * sin(dv);
      Eigen::Vector3d ep = Eigen::Vector3d(x, y, z);
      frustum_endpoints.push_back(ep);
      // Eigen::Vector3d ep_B = rot_B2S * ep + center_offset;
      Eigen::Vector3d ep_B = ep;
      frustum_endpoints_B.push_back(ep_B);
    }
    if (width == 0) {
      width = w;
    }
  }
  height = h;
  ROS_INFO_STREAM("Computed multiray_endpoints for volumetric gain [kCamera]:" 
                  << frustum_endpoints_B.size() << " points");
}

bool TsdfServer::checkUnknownStatus(
    const voxblox::TsdfVoxel* voxel) const {
  if (voxel == nullptr || voxel->weight < 1e-6) {
    return true;
  }
  return false;
}

float TsdfServer::getScanStatus(
    Eigen::Vector3d& pos, std::vector<Eigen::Vector3d>& multiray_endpoints,
    std::tuple<int, int, int>& gain_log,
    std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& voxel_log,
    SensorParamsBase& sensor_params) {
  unsigned int num_unknown_voxels = 0, num_free_voxels = 0,
               num_occupied_voxels = 0;
  float interestingness = 0.0f;
  const float voxel_size = sdf_layer_->voxel_size();
  const float voxel_size_inv = 1.0 / voxel_size;
  // const float step_size_change_dist =
  //     4.0;  // After every these many meters increase the ray interpolation
  //           // distance
  // const float step_size_change_dist_end =
  //     12.0;  // After this much distance stick to the last interpolation
  //            // distance

  const voxblox::Point start_scaled =
      pos.cast<voxblox::FloatingPoint>() * voxel_size_inv;
  // const float distance_thres =
      // occupancy_distance_voxelsize_factor_ * sdf_layer_->voxel_size() + 1e-6;
  const float distance_thres = voxel_size + 1e-6;

  // NOTES: no optimization / no twice-counting considerations possible without
  // refactoring planning strategy here

  // Iterate for every endpoint, insert unknown voxels found over every ray into
  // a set to avoid double-counting Important: do not use <VoxelIndex> type
  // directly, will break voxblox's implementations
  // Move away from std::unordered_set and work with std::vector + std::unique
  // count at the end (works best for now)
  /*std::vector<std::size_t> raycast_unknown_vec_, raycast_occupied_vec_,
  raycast_free_vec_; raycast_unknown_vec_.reserve(multiray_endpoints.size() *
  tsdf_integrator_config_.max_ray_length_m * voxel_size_inv); //optimize for
  number of rays raycast_free_vec_.reserve(multiray_endpoints.size() *
  tsdf_integrator_config_.max_ray_length_m * voxel_size_inv); //optimize for
  number of rays raycast_occupied_vec_.reserve(multiray_endpoints.size());
  //optimize for number of rays*/
  // voxel_log.reserve(multiray_endpoints.size() *
  // tsdf_integrator_config_.max_ray_length_m * voxel_size_inv); //optimize for
  // number of rays
  Point origin;
  origin << 0.0, 0.0, 0.0;
  TsdfIntegratorBase::Config integrator_config = tsdf_integrator_->getConfig();
  voxblox::GlobalIndex global_index;
  for (size_t i = 0; i < multiray_endpoints.size(); ++i) {
    // float step_size = voxel_size;//* ray_cast_step_size_multiplier_;
    // float step_size_inv = 1.0 / step_size;
    // float og_step_size = step_size;
    // Eigen::Vector3d ray_normalized =
    //     (multiray_endpoints[i] - pos);  // Not yet noramlized
    // double ray_norm = ray_normalized.norm();
    // ray_normalized = ray_normalized / ray_norm;  // Normalized here
    // // Iterate over the ray.
    // double prev_step_dist = 0.0;
    // for (double step = 0.0; step <= ray_norm; step += step_size) { // FIX IT !!!!!!!!!!!!!!!!
    //   Eigen::Vector3d voxel_coordi = (pos + ray_normalized * step);
    //   voxblox::GlobalIndex global_index =
    //       voxblox::getGridIndexFromPoint<voxblox::GlobalIndex>(
    //           voxel_coordi.cast<voxblox::FloatingPoint>(), voxel_size_inv);
    //   voxblox::TsdfVoxel* voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index);
    //   // Unknown
    //   if (checkUnknownStatus(voxel)) {
    //     /*raycast_unknown_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
    //     ++num_unknown_voxels;
    //     if (voxel != nullptr) {
    //       interestingness += voxel->interestingness; // COUNT INTERESTINGNESS OF UNKNOWN VOXEL
    //     }
    //     voxel_log.push_back(std::make_pair(
    //         voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
    //             .cast<double>(),
    //         VoxelStatus::kUnknown));
    //     continue;
    //   }
    //   // Free
    //   if (voxel->distance > distance_thres) {
    //     /*raycast_free_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
    //     ++num_free_voxels;
    //     voxel_log.push_back(std::make_pair(
    //         voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
    //             .cast<double>(),
    //         VoxelStatus::kFree));
    //     continue;
    //   }
    //   // Occupied
    //   /*raycast_occupied_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
    //   ++num_occupied_voxels;
    //   // interestingness += voxel->interestingness; // COUNT INTERESTINGNESS OF OCCUPIED VOXEL
    //   voxel_log.push_back(std::make_pair(
    //       voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
    //           .cast<double>(),
    //       VoxelStatus::kOccupied));
    //   break;
    // }
  
    RayCaster ray_caster(origin, multiray_endpoints[i].cast<voxblox::FloatingPoint>(), false,
                         integrator_config.voxel_carving_enabled,
                         integrator_config.max_ray_length_m, voxel_size_inv,
                         integrator_config.default_truncation_distance, true);
    while (ray_caster.nextRayIndex(&global_index)) {
      voxblox::TsdfVoxel* voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index);
      // Unknown
      if (checkUnknownStatus(voxel)) {
        /*raycast_unknown_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
        ++num_unknown_voxels;
        if (voxel != nullptr) {
          if (!voxel->is_interestingness_counted) {
            interestingness += voxel->interestingness; // COUNT INTERESTINGNESS OF UNKNOWN VOXEL
            voxel->is_interestingness_counted = true;
            observed_interesting_unknown_voxels->push_back(global_index);
          }
        }
        // voxel_log.push_back(std::make_pair(
        //     voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
        //         .cast<double>(),
        //     VoxelStatus::kUnknown));
        continue;
      }
      // Free
      if (voxel->distance > distance_thres) {
        /*raycast_free_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
        ++num_free_voxels;
        // voxel_log.push_back(std::make_pair(
        //     voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
        //         .cast<double>(),
        //     VoxelStatus::kFree));
        continue;
      }
      // Occupied
      /*raycast_occupied_vec_.push_back(std::hash<voxblox::GlobalIndex>()(global_index));*/
      ++num_occupied_voxels;
      // interestingness += voxel->interestingness; // COUNT INTERESTINGNESS OF OCCUPIED VOXEL
      // voxel_log.push_back(std::make_pair(
      //     voxblox::getCenterPointFromGridIndex(global_index, voxel_size)
      //         .cast<double>(),
      //     VoxelStatus::kOccupied));
      break;
    }        
  }
  /*std::sort(raycast_unknown_vec_.begin(), raycast_unknown_vec_.end());
  std::sort(raycast_occupied_vec_.begin(), raycast_occupied_vec_.end());
  std::sort(raycast_free_vec_.begin(), raycast_free_vec_.end());
  num_unknown_voxels = std::unique(raycast_unknown_vec_.begin(),
  raycast_unknown_vec_.end()) - raycast_unknown_vec_.begin();
  num_occupied_voxels = std::unique(raycast_occupied_vec_.begin(),
  raycast_occupied_vec_.end()) - raycast_occupied_vec_.begin();
  num_free_voxels = std::unique(raycast_free_vec_.begin(),
  raycast_free_vec_.end()) - raycast_free_vec_.begin();*/
  gain_log =
      std::make_tuple(num_unknown_voxels, num_free_voxels, num_occupied_voxels);
  return interestingness;
}

void TsdfServer::computeVolumetricGainRayModelNoBound(StateVec& state,
                                               VolumetricGain& vgain) {
  vgain.reset();

  // std::vector<std::tuple<int, int, int>> gain_log;
  std::vector<std::pair<Eigen::Vector3d, VoxelStatus>> voxel_log;
  // @TODO tung.
  // Compute for each sensor in the exploration sensor list.
  // However, this would be a problem if those sensors have significant overlap.


  Eigen::Vector3d origin(state[0], state[1], state[2]);
  std::tuple<int, int, int> gain_log_tmp;
  std::vector<std::pair<Eigen::Vector3d, VoxelStatus>> voxel_log_tmp;
  std::vector<Eigen::Vector3d> multiray_endpoints;
  camera_param.getFrustumEndpoints(state, multiray_endpoints);
  float interestingness = getScanStatus(origin, multiray_endpoints, gain_log_tmp,
                              voxel_log_tmp,
                              camera_param);
  // int num_unknown_voxels = 0, num_free_voxels = 0, num_occupied_voxels = 0;
  // Have to remove those not belong to the local bound.
  // At the same time check if this is frontier.

  // DO WE NEED THIS ???????
  // for (auto& vl : voxel_log_tmp) {
  //   Eigen::Vector3d voxel = vl.first;
  //   VoxelStatus vs = vl.second;
  //   if (vs == VoxelStatus::kUnknown) {
  //     ++num_unknown_voxels;
  //   } else if (vs == VoxelStatus::kFree) {
  //     ++num_free_voxels;
  //   } else if (vs == VoxelStatus::kOccupied) {
  //     ++num_occupied_voxels;
  //   } else {
  //     ROS_ERROR("Unsupported voxel type.");
  //   }
  // }

  // ROS_INFO_STREAM("num_unknown_voxels:" << num_unknown_voxels
  //                 << ", num_free_voxels:" << num_free_voxels
  //                 << ", num_occupied_voxels:" << num_occupied_voxels);

  vgain.num_unknown_voxels = std::get<0>(gain_log_tmp);//num_unknown_voxels;
  vgain.num_free_voxels = std::get<1>(gain_log_tmp);//num_free_voxels;
  vgain.num_occupied_voxels = std::get<2>(gain_log_tmp);//num_occupied_voxels;  
  vgain.gain = interestingness;
}

void TsdfServer::spreadInterestingness(GlobalIndex global_index) {
  voxblox::TsdfVoxel* voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index);
  CHECK_NOTNULL(voxel);

  AlignedQueue<GlobalIndex> voxel_queue;
  voxel_queue.push(global_index);
  
  while (!voxel_queue.empty()) {
    // Get the global indices of neighbors.
    const GlobalIndex global_index_tmp = voxel_queue.front();
    voxel_queue.pop();
    voxblox::TsdfVoxel* parent_voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index_tmp);
    Neighborhood<>::IndexMatrix neighbor_indices;
    Neighborhood<>::getFromGlobalIndex(global_index_tmp, &neighbor_indices);

    // Go through the neighbors and see if we can update any of them.
    for (unsigned int idx = 0u; idx < neighbor_indices.cols(); ++idx) {
      const GlobalIndex& neighbor_index = neighbor_indices.col(idx);
      voxblox::TsdfVoxel* neighbor_voxel = sdf_layer_->getVoxelPtrByGlobalIndex(neighbor_index);        
      if (neighbor_voxel == nullptr) { // can miss some unknown voxels here!
        continue;
      }
      // update interesting level if this's unknown voxel and need to be updated
      if (checkUnknownStatus(neighbor_voxel)) {
        if (neighbor_voxel->interesting_distance > parent_voxel->interesting_distance + 1) {
          neighbor_voxel->interesting_distance = parent_voxel->interesting_distance + 1;
          neighbor_voxel->interestingness = decay_lambda_ * parent_voxel->interestingness; // decay function
        }
        // stop the spreading when the voxel is too far away from the original interesting voxels
        if (!neighbor_voxel->in_queue &&
            (neighbor_voxel->interesting_distance < decay_distance_)) {  
          voxel_queue.push(neighbor_index);
          neighbor_voxel->in_queue = true;
        }
      }
    }
  }
}

bool TsdfServer::calcInfoGainCallback(voxblox_msgs::InfoGain::Request& request,
                                      voxblox_msgs::InfoGain::Response& response) {  // NOLINT
  StateVec state_vec;
  int32_t num_cam_poses = request.camera_poses.size() / 6;
  int32_t idx = 0;
  response.info_gain.reserve(num_cam_poses);
  // integrate the pcl
  // auto start = std::chrono::steady_clock::now();
  sensor_msgs::PointCloud2::Ptr pcl_in = boost::make_shared<sensor_msgs::PointCloud2>(request.pcl);
  lightInsertPointcloud(pcl_in, interesting_voxels);
  
  // spread interestingness out
  for (idx = 0; idx < interesting_voxels->size(); idx++) {
    // get the global index by coordinate
    // get the voxel, set interesting_distance = 0
    spreadInterestingness(interesting_voxels->at(idx));
  }

  if (publish_pointclouds_on_update_) {
    publishPointclouds();
  }

  // calculate info gain
  // auto end1 = std::chrono::steady_clock::now();
  for (idx = 0; idx < 6 * num_cam_poses; idx = idx + 6) {
    // calculate the gain by ray castingsolve_time_average_
    state_vec << request.camera_poses[idx], request.camera_poses[idx + 1],
                 request.camera_poses[idx + 2], request.camera_poses[idx + 3],
                 request.camera_poses[idx + 4], request.camera_poses[idx + 5];
    computeVolumetricGainRayModelNoBound(state_vec, vgain);
    response.info_gain.push_back(vgain.gain);
    // reset observed_interesting_unknown_voxels for next frame
    for (int32_t idx2 = 0; idx2 < observed_interesting_unknown_voxels->size(); idx2++) {
      voxblox::GlobalIndex global_index = observed_interesting_unknown_voxels->at(idx2);
      voxblox::TsdfVoxel* voxel = sdf_layer_->getVoxelPtrByGlobalIndex(global_index);
      voxel->is_interestingness_counted = false;
    }
    observed_interesting_unknown_voxels->clear();
  }
  
  // clear the map
  // auto end2 = std::chrono::steady_clock::now();
  clear();
  // auto end3 = std::chrono::steady_clock::now();
  // std::chrono::duration<double> elapsed_seconds_integrate = end1 - start;
  // std::chrono::duration<double> elapsed_seconds_info_gain = end2 - end1;
  // std::chrono::duration<double> elapsed_seconds_clear = end3 - end2;
  // std::cout << "INTEGRATE time: " << elapsed_seconds_integrate.count() << "s\n";
  // std::cout << "INFO_GAIN time: " << elapsed_seconds_info_gain.count() << "s\n";
  // std::cout << "CLEAR time: " << elapsed_seconds_clear.count() << "s\n";
  return true;
}

}  // namespace voxblox
