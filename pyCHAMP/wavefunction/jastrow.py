import numpy as np

@staticmethod
def pairwise_distance(pos,pos2=None):
	'''Returns the pairwise distance of all electrons.
	Args:
		pos : nd array shape (N,3) : positions of the electorns
		pos2 : nd array sahpe (N,3) : positions of the electorns
	Returns
		dist : nd array shape (N,N)
		dist[i,j] = || pos[i,:]-pos[:,j] ||
	'''
	x = pos
	if pos2 is None:
		y = x
	else:
		y = x2 
	x2 = np.sum(x**2,1).reshape(-1,1)
	y2 = np.sum(y**2,1).reshape(1,-1)
	return x2 + y2 - 2.0 * np.dot(x,y.T)

def _jast(pos,b,bp=1,pos2=None):
	Rij = pairwise_distance(pos,pos2)
	return np.exp(np.sum(b*Rij / (1+bp*Rij)))

def jastrow(pos_up,pos_down):
	return _jast(pos_up,b=0.25) * _jast(pos_down,b=0.25) * _jast(pos_up,b=0.5,pos2=pos_down)