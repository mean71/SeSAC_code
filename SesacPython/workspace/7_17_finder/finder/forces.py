from particles import Particle
from util import *
    #animation에서만 호출이 보이는 두가지 함수파일
def gravitational_force(source, target):        #animation에서 호출
    '''
    calculate force from source particle to target particle
    '''
    m_s = source.properties['mass']
    m_t = target.properties['mass']
    G = 1
    mag = G * m_s * m_t / dist(source.position, target.position)    # 11,12 util.py에서 가져온 함수
    r_hat = dist_unitvec(source.position, target.position)
    return mag * r_hat[0], mag * r_hat[1]
    
def restoring_force(source, target):                #
    k = 0.01
    return (k * (source.position[0] - target.position[0]), 
            k * (source.position[1] - target.position[1]))
    