# -*- coding: UTF-8 -*-
import numpy as np
import util
import glob

'''
keep z-axis unchanged
p = log(sqrt(x^2+ y^2)
theta = arctan^2(y/x
'''

def self_log(x, base):
    return np.log(x) / np.log(base)


def cart2polar(input_xyz):
    '''
    r = sqrt(x^2+y^2)
    phi = atan2(y/x)
    '''
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def cart2polar_adv(input_xyz):
    '''
    r = sqrt(x^2+y^2)
    theta = atan2(y/x)
    理论上，计算r>0, phi\in (-pi, pi], atan2存在多种情况
    简化的版本利用先计算r，使用替代可以简化计算phi
    1. arccos(x/r) if y>=0 and r!=0
    2. -arccos(x/r) if y<0
    3. undefined r=0
    实际计算前，先剔除0值
    但是结果和cart2polar() 一样，没差别误差在小数点后11位
    所以推断，下面的logpolar应该也不受影响
    '''
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.where(input_xyz[:, 1] < 0, -np.arccos(input_xyz[:, 0] / rho), np.arccos(input_xyz[:, 0] / rho))
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def cart2logpolar(input_xyz):
    #rho = self_log(np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2), 2)  # base=e
    '''
    先用自然指数e做底
    r=ln(sqrt(x^2+y^2))
    phi = atan2(y,x)
    '''
    rho = np.log(np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)) # base=e
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def test(xyz):
    x = xyz[:, 0].T
    y = xyz[:, 1].T
    rho = np.log(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)) # base=e
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    xx = np.exp(rho) * np.cos(phi)
    yy = np.exp(rho) * np.sin(phi)
    a = x - xx
    b = y - yy
    print(a, b)


def polar2cat(input_xyz_polar):
    '''
    x = r * cos(phi)
    y = r * sin(phi)
    '''
    x = input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])
    y = input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])
    return np.stack((x, y, input_xyz_polar[:, 2]), axis=1)

def logpolar2cat(input_xyz_logpolar):
    '''
    x = e^p cos(phi)
    y = e^p sin(phi)
    '''
    x = np.exp(input_xyz_logpolar[:, 0]) * np.cos(input_xyz_logpolar[:, 1])
    y = np.exp(input_xyz_logpolar[:, 0]) * np.sin(input_xyz_logpolar[:, 1])
    return np.stack((x, y, input_xyz_logpolar[:, 2]), axis=1)


if __name__ == '__main__':

    seq10 = './seq10' # 'D:/ov/sequences/00'  # './seq10'
    pcl = glob.glob(seq10 + "/*.bin")
    labell = glob.glob(seq10 + "/*.label")
    seq_len = len(pcl)
    color_map, remap = util.load_colormap('./data/semantic-kitti.yaml')
    for i in range(0, seq_len):
        fpc = pcl[i]  # pc
        flabel = labell[i]  # label
        pc = util.load_pclabel(fpc, flabel, remap)  # [x,y,z,r,l]

        #xyz_polar = cart2polar(pc[:, :3])
        # xyz_polar1 = cart2polar_adv(pc[:, :3])
        #xyz = polar2cat(xyz_polar)
        #aaa = xyz - pc[:, :3]
        #print(aaa)
        test(pc[:, :3])
        xyz_logpolar = cart2logpolar(pc[:, :3])
        xyz1 = logpolar2cat(xyz_logpolar)
        bbb = xyz1 - xyz_logpolar
        print(bbb)
