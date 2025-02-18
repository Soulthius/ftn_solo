import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ftn_solo.tasks.robot_squat import RobotMove


class TwistModifierNode(Node):
    def __init__(self):
        super().__init__('movement_logic')

        self.task = RobotMove()
        # Subscriber to the /cmd_vel topic
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10
        )
        self.subscription  # prevent unused variable warning

      
    def twist_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.linear.x)
        self.task.define_movement(msg)

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    movement_logic = TwistModifierNode()

    # Spin the node
    rclpy.spin(movement_logic)

    # Destroy the node explicitly (optional)
    movement_logic.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
