#!/bin/python3
import torch
import torchvision

def get_entry_point():
    return 'ImageAgent'


def Learning_by_cheating(tick_data,pos,turn_controller,speed_controller,net,converter):
    """
    Learning By Cheating controller code
    """
    img = torchvision.transforms.functional.to_tensor(tick_data['image'])
    img = img[None].cuda()

    target = torch.from_numpy(tick_data['target'])
    target = target[None].cuda()

    points, (target_cam, _) = self.net.forward(img, target)
    points_cam = points.clone().cpu()
    points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
    points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
    points_cam = points_cam.squeeze()
    points_world = self.converter.cam_to_world(points_cam).numpy()

    aim = (points_world[1] + points_world[0]) / 2.0
    angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
    steer = turn_controller.step(angle)
    steer = np.clip(steer, -1.0, 1.0)

    desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
    # desired_speed *= (1 - abs(angle)) ** 2

    speed = tick_data['speed']
    theta = tick_data['compass']

    brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

    delta = np.clip(desired_speed - speed, 0.0, 0.25)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, 0.75)
    throttle = throttle if not brake else 0.0


    return steer,speed,brake
