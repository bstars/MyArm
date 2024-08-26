
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt 
import torch
import cvxpy

class MyArm:
    def __init__(self) -> None:
        self.ls = np.array([6., 8.5, 8., 5.5]) # in cm
        self.thetas_tilde = np.array([0., 0., 0., 0.]) # joint angles

        # joint angle limits
        # note that the last angle is for the end-effector grasping
        # so it's not related to the kinematics
        self.lbs = np.array([   -np.pi/2,   -np.pi/6,       -1/12 * np.pi,  np.deg2rad(-1)])
        self.ubs = np.array([   np.pi/2,    np.pi/2 * 0.8,    1/3 * np.pi,    np.deg2rad(8)])

        # T_adjs[i] is the transformation matrix from frame {i} to {i+1}
        self.T_adjs = [None, None, None]

        # T_0i[i] is the transformation matrix from frame {0} to {i}
        self.T_0i = [np.eye(4), None, None, None]


        # initialize forward kinematics
        self.forward()

        # plotting initialize
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_aspect('auto')

    @staticmethod
    def transformation_matrix(theta, alpha, r, d, lib=np):
        """
        Construct transformation matrix using Denavit-Hartenberg Frame
        """
        if lib==np:
            return np.array([
                [ np.cos(theta), -np.cos(alpha) * np.sin(theta), np.sin(alpha) * np.sin(theta),  r * np.cos(theta) ],
                [ np.sin(theta), np.cos(alpha) * np.cos(theta),  -np.sin(alpha) * np.cos(theta), r * np.sin(theta) ],
                [0,              np.sin(alpha),                  np.cos(alpha),                   d],
                [0,               0,                              0,                               1]
            ])
        
        elif lib==torch:
            temp = torch.zeros((4,4))
            temp[0,0] = torch.cos(theta)
            temp[0,1] = -np.cos(alpha) * torch.sin(theta)
            temp[0,2] = np.sin(alpha) * torch.sin(theta)
            temp[0,3] = r * torch.cos(theta)

            temp[1,0] = torch.sin(theta)
            temp[1,1] = np.cos(alpha) * torch.cos(theta)
            temp[1,2] = -np.sin(alpha) * torch.cos(theta)
            temp[1,3] = r * torch.sin(theta) 

            temp[2,1] = np.sin(alpha)
            temp[2,2] = np.cos(alpha)
            temp[2,3] = d

            temp[3,3] = 1

            return temp
        
    def forward(self):
        """
        compute the forward kinematics with Denavit-Hartenberg frame
        this function should be called everytime the angles are updated
        """

                                                #       theta                           alpha       r                   d
        self.T_adjs[0] = self.transformation_matrix(self.thetas_tilde[0],            -np.pi/2,      0,              self.ls[0])
        self.T_adjs[1] = self.transformation_matrix(self.thetas_tilde[1] - np.pi/2,     0,          self.ls[1],     0)
        self.T_adjs[2] = self.transformation_matrix(self.thetas_tilde[2] + np.pi/2,     0,          self.ls[2],     0)

        self.T_0i = accumulate(
            iterable=self.T_adjs,
            func=lambda A,B: A @ B,
            initial=np.eye(4) # initial value is the trivial transformation from frame {0} to {0}
        )
        self.T_0i = list(self.T_0i)

    def render(self, pause=False):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

        # origin of all frames
        origins = np.stack([
            T[0:3, -1] for T in self.T_0i
        ])

        claw = origins[-1].copy()
        temp = self.T_0i[-1][0:2,-1]
        claw[:2] += temp / np.linalg.norm(temp) * self.ls[-1]

        origins = np.concatenate([origins, claw[None,:]], axis=0)


        # plot the joints and links
        self.ax.plot3D(origins[:,0], origins[:,1], origins[:,2])
        self.ax.scatter3D(origins[:,0], origins[:,1], origins[:,2])

        # plot the frame for end-effector
        colors = ['r', 'g', 'b']
        for i in range(3):

            temp = np.concatenate([
                self.T_0i[-1][0:3,-1],self.T_0i[-1][0:3,i]*2
            ])
            self.ax.quiver(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], color=colors[i])


        self.ax.set_xlim3d([0, 20])
        self.ax.set_ylim3d([-20, 20])
        self.ax.set_zlim3d([0, 20])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_aspect("auto")

        if pause is False:
            plt.pause(0.001)
        else:
            plt.show()

    def IK_loss_function(self, thetas, target):
        """
        compute the loss function (and gradient) in Inverse kinematics
        the result will be used for inverse kinematics by PGD or SQP
        """

        T_adjs = [None, None, None]

                                                #  theta              alpha         r                   d
        T_adjs[0] = self.transformation_matrix(thetas[0],            -np.pi/2,      0,              self.ls[0], lib=torch)
        T_adjs[1] = self.transformation_matrix(thetas[1] - np.pi/2,     0,          self.ls[1],     0, lib=torch)
        T_adjs[2] = self.transformation_matrix(thetas[2] + np.pi/2,     0,          self.ls[2],     0, lib=torch)

        T_0i = accumulate(
            iterable=T_adjs,
            func=lambda A,B: A @ B,
            initial=torch.eye(4) # initial value is the trivial transformation from frame {0} to {0}
        )
        T_0i = list(T_0i)

        claw = T_0i[-1][0:3, -1]
        temp = T_0i[-1][0:2,-1].clone()
        claw[:2] += temp / torch.norm(temp) * self.ls[-1]

        loss = torch.sum((claw - target)**2)
        
        return loss
        
    
    def IK_SQP(self, target:np.array, alpha=0.5, beta=0.5):
        """
        Compute the inverse kinematics using a home-made Sequential Quadratic Programming (SQP) algorithm with GD-like convergence rate
        it's especially suitable for L-smooth constraints with small Lipschitz constant.

        I'm currently not willing to expose too much about the algorithm, you can replace it with Projected Gradient Descent(PGD)

        Also, in this case GD converges faster in wall-clock time and generate smoother path
        """
        target_torch = torch.from_numpy(target).float()
        thetas = self.thetas_tilde.copy()

        while True:
            thetas_torch = torch.from_numpy(thetas).float()
            thetas_torch.requires_grad = True
            obj_f0 = self.IK_loss_function(
                thetas_torch, target_torch
            )
            thetas_torch.retain_grad()
            obj_f0.backward()
            g_f0 = thetas_torch.grad.detach().numpy()
            obj_f0 = obj_f0.item()
            

            # construct the QP subproblem
            u = cvxpy.Variable([len(self.thetas_tilde), ])
            v = cvxpy.Variable([1,])


            constraints = [g_f0 @ u <= v[0]]
            for i in range(len(self.thetas_tilde)-1):
                constraints.extend( [self.lbs[i] - thetas[i] - u[i] <= v[0] ])
                constraints.extend( [thetas[i] - self.ubs[i] + u[i] <= v[0] ])

            
            qp = cvxpy.Problem(
                cvxpy.Minimize(
                    0.5 * cvxpy.sum(cvxpy.square(u)) + v[0]
                ),
                constraints=constraints
            )

            qp.solve(solver=cvxpy.MOSEK)

            
            u = u.value
            v = v.value[0]

            if np.abs(v) <= 1e-2:
                break

            t = 1

            # line search for feasibility
            while np.any( (thetas + t * u <= self.lbs)[:-1] ) \
                or np.any( (thetas + t * u >= self.ubs)[:-1] ):
                t *= beta
                
            while self.IK_loss_function(
                torch.from_numpy(thetas + t * u).float(), 
                target_torch
            ).item() > obj_f0 + alpha * t * u @ g_f0:
                t *= beta

            thetas += t * u
            self.thetas_tilde = thetas.copy()
            self.forward()
            yield obj_f0


    def IK_PGD(self, target):
        """
        Compute inverse kinematics with Projected Gradient Descent
        """
        target_torch = torch.from_numpy(target).float()
        thetas_torch = torch.from_numpy(self.thetas_tilde).float()
        thetas_torch.requires_grad = True
        optimizer = torch.optim.Adam([thetas_torch], lr=2e-2)
        lbs = torch.from_numpy(self.lbs).float()
        ubs = torch.from_numpy(self.ubs).float()
        obj_prev = np.inf

        while True:
            self.IK_loss_function( thetas_torch, target_torch )
            loss = self.IK_loss_function( thetas_torch, target_torch )

            optimizer.zero_grad()
            loss.backward()
            g = thetas_torch.grad.detach().numpy()
            optimizer.step()

            if np.linalg.norm(g) <= 1e-2 or loss.item() < 0.01:
                break

            if np.abs(obj_prev - loss.item()) < 1e-5:
                break
            
            thetas_torch.data = torch.clamp(thetas_torch.data, min=lbs, max=ubs)
            self.thetas_tilde = thetas_torch.clone().detach().numpy()
            self.forward()
            obj_prev = loss.item()
            yield loss.item()
        


            


if __name__ == '__main__':

    arm = MyArm()
    arm.forward()
    # arm.render(pause=True)


    # for i, obj in enumerate(arm.IK_SQP(np.array([10, 2, 8.]))):
    #     print(obj)

    #     arm.render(pause=False)
       

    for obj in arm.IK_PGD(np.array([12, 0, 5.])):
        print(obj)

        arm.render(pause=False)


   

