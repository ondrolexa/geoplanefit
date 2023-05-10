import numpy as np
from scipy.optimize import minimize_scalar
from apsg import vecset, vec, fol


class GeoPlane:
    """Plane fitting class."""

    def __init__(self, coords, crs):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.crs = crs
        self.coords = np.asarray(coords)
        if len(self.coords) > 3:
            res = minimize_scalar(self.scaling, bounds=(1, 200), method='bounded', args=(False,), options=dict(maxiter=100))
            self.scale = res.x
            self.ot = self.scaling(self.scale, True)
            fs = self.ot.eigenfols[2]
            self.fol = fol(fs.x, fs.y, self.scale * fs.z).normalized()
        else:
            points = np.array([self.coords.T[1],
                               self.coords.T[0],
                               self.coords.T[2]
                               ])
            # shift to centre
            ctr = points.mean(axis=1)
            x = points - ctr[:, np.newaxis]
            # get ot
            self.scale = 1
            self.ot = vecset([vec(v) for v in x.T]).ortensor()
            fs = self.ot.eigenfols[2]
            self.fol = fol(fs.x, fs.y, fs.z).normalized()

    @property
    def coords_geo_scaled(self):
        return np.array([self.coords.T[1],
                         self.coords.T[0],
                         -self.scale * self.coords.T[2]
                         ]).T

    @property
    def coords_geo(self):
        return np.array([self.coords.T[1],
                         self.coords.T[0],
                         -self.coords.T[2]
                         ]).T

    def to_dict(self):
        """Object serialization"""
        return dict(crs=self.crs, coords=self.coords.tolist())

    def __repr__(self):
        if len(self.coords) > 3:
            return f'{self.fol} [{len(self.coords)}] K: {self.ot.k:6.4f} dz: {max(self.coords.T[2]) - min(self.coords.T[2]):.2f}'
        else:
            return f'{self.fol} [{len(self.coords)}] R: {self.ot.Rxy:6.4f} dz: {max(self.coords.T[2]) - min(self.coords.T[2]):.2f}'

    def scaling(self, scale, get_tensor):
        points = np.array([self.coords.T[1],
                           self.coords.T[0],
                           -scale * self.coords.T[2]
                           ])
        # shift to centre
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        # get ot
        ot = vecset([vec(v) for v in x.T]).ortensor()
        if get_tensor:
            return ot
        else:
            return ot.k
