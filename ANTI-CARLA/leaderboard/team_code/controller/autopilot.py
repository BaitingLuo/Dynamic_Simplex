#!/bin/python3

def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def autopilot(self,target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed
        #far_node, far_command = self._command_planner.run_step(gps)

def _get_angle_to(self, pos, theta, target):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle
    return angle

def _should_brake(self):
    actors = self._world.get_actors()

    vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
    light = self._is_light_red(actors.filter('*traffic_light*'))
    walker = self._is_walker_hazard(actors.filter('*walker*'))

    return any(x is not None for x in [vehicle, light, walker])

def _is_light_red(self, lights_list):
    if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
        affecting = self._vehicle.get_traffic_light()

        for light in self._traffic_lights:
            if light.id == affecting.id:
                return affecting

    return None

def _is_walker_hazard(self, walkers_list):
    z = self._vehicle.get_location().z
    p1 = _numpy(self._vehicle.get_location())
    v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

    self._draw_line(p1, v1, z+2.5, (0, 0, 255))

    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat

        self._draw_line(p2, v2, z+2.5)

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            return walker

    return None


def _is_vehicle_hazard(self, vehicle_list):
    z = self._vehicle.get_location().z

    o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
    p1 = _numpy(self._vehicle.get_location())
    s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
    v1_hat = o1
    v1 = s1 * v1_hat

    self._draw_line(p1, v1, z+2.5, (255, 0, 0))

    for target_vehicle in vehicle_list:
        if target_vehicle.id == self._vehicle.id:
            continue

        o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
        p2 = _numpy(target_vehicle.get_location())
        s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
        v2_hat = o2
        v2 = s2 * v2_hat

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        self._draw_line(p2, v2, z+2.5, (255, 0, 0))

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        return target_vehicle

    return None

def _draw_line(self, p, v, z, color=(255, 0, 0)):
    if not DEBUG:
        return

    p1 = _location(p[0], p[1], z)
    p2 = _location(p[0]+v[0], p[1]+v[1], z)
    color = carla.Color(*color)

    #self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

def set_global_plan(self, global_plan_gps, global_plan_world_coord):
    super().set_global_plan(global_plan_gps, global_plan_world_coord)
    self._plan_HACK = global_plan_world_coord
    self._plan_gps_HACK = global_plan_gps
