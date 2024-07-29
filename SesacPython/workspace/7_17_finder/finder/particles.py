# assume cartesian coordinate
# 추정하다 데카르트? 좌표를
class Particle:
    def __init__(self, particle_id, position, velocity, properties):
        assert isinstance(particle_id, int)
        self.particle_id = particle_id
        
        assert len(position) == 2
        assert len(velocity) == 2
        assert isinstance(position[0], float)
        assert isinstance(position[1], float)
        assert isinstance(velocity[0], float)
        assert isinstance(velocity[1], float)
        
        self.position = position 
        self.velocity = velocity 
        
        assert isinstance(properties, dict)
        assert 'mass' in properties.keys()
        assert isinstance(properties['mass'], float), properties['mass']
        self.properties = properties 
        
    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        return self.particle_id == other.particle_id
    def __neq__(self, other):
        return not self.__eq__(other)