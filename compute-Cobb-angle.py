from skimage import measure
from tiger.io import read_image, write_image
from tiger.resampling import reorient_image, resample_mask
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import ast


def PCA(data, correlation=False, sort=True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix.

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:, 2]

    #: get a point from the plane
    point = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
    else:
        return point, normal


def plane_to_tangent(vertices, point, normal):
    x = range(int(round(vertices[:, 0].min())), int(round(vertices[:, 0].max())))
    d = -point[[0, 2]].dot(normal[[0, 2]])
    z = (-normal[0] * x - d) * 1. / normal[2]
    return x, z


class Disc:
    def __init__(self, mask):
        verts, faces, normals, values = measure.marching_cubes(mask_temp, 0.5)
        self.faces = faces
        self.normals = normals
        self.verts = verts
        self.upper_plane()
        self.lower_plane()


    def vertices_to_plane_surface(self, vertices):
        point, normal = best_fitting_plane(vertices)
        xx_range = range(int(round(vertices[:, 0].min())), int(round(vertices[:, 0].max())))
        yy_range = range(int(round(vertices[:, 1].min())), int(round(vertices[:, 1].max())))
        xx, yy = np.meshgrid(xx_range, yy_range)
        d = -point.dot(normal)
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        if normal[2] < 0:
            normal *= -1

        return xx, yy, z, point, normal


    def split_disc(self, vertices):
        point, normal = best_fitting_plane(vertices)
        d = -point.dot(normal)
        y = (-normal[0] * vertices[:, 0] - normal[1] * vertices[:, 1] - d) * 1. / normal[2]
        difference = vertices[:, 2] - y
        self.upper = difference > 0
        self.lower = difference < 0
        self.xx_M, self.yy_M, self.z_M, self.point_M, self.normal_M = self.vertices_to_plane_surface(vertices)
        return self.upper, self.lower


    def upper_plane(self):
        upper, lower = self.split_disc(self.verts)
        self.xx_U, self.yy_U, self.z_U, self.point_U, self.normal_U = self.vertices_to_plane_surface(self.verts[upper, :])
        self.angle_U = np.rad2deg(np.arctan(self.normal_U[0] / self.normal_U[2]))


    def lower_plane(self):
        upper, lower = self.split_disc(self.verts)
        self.xx_L, self.yy_L, self.z_L, self.point_L, self.normal_L = self.vertices_to_plane_surface(self.verts[lower, :])
        self.angle_L = np.rad2deg(np.arctan(self.normal_L[0] / self.normal_L[2]))


    def to_dict(self):
        return {
            'verts': self.verts.tolist(),
            'faces': self.faces.tolist(),
            'xx_L': self.xx_L.tolist(),
            'xx_U': self.xx_U.tolist(),
            'xx_M': self.xx_M.tolist(),
            'yy_L': self.yy_L.tolist(),
            'yy_U': self.yy_U.tolist(),
            'yy_M': self.yy_M.tolist(),
            'z_L': self.z_L.tolist(),
            'z_U': self.z_U.tolist(),
            'z_M': self.z_M.tolist(),
            'point_L': self.point_L.tolist(),
            'point_U': self.point_U.tolist(),
            'point_M': self.point_U.tolist(),
            'normal_L': self.normal_L.tolist(),
            'normal_U': self.normal_U.tolist(),
            'normal_M': self.normal_U.tolist(),
            'upper': self.upper.tolist(),
            'lower': self.lower.tolist()
        }



#%%
mask_path = Path('path/to/spine/segmentations')
image_path = Path('path/to/output/images')
all_masks = sorted(list(mask_path.glob('*.mha')))
all_scans = dict()
angles_manual = []
index = 0
total_discs = 0
total_patients = 0

largest_angle = True

#%%
for file in all_masks:#mask_path.iterdir():
      total_patients += 1
      mask, header = read_image(file)
      mask, header = reorient_image(mask, header, interpolation='nearest')
      mask = resample_mask(mask, header.spacing, (0.5, 0.5, 0.5))
      mask_vertebrae = mask.copy()
      mask[mask < 200] = 0
      labels = sorted(list(np.unique(mask[mask > 0])))
      # mask[mask > 0] = 1
      try:
          verts_all, faces_all, normals_all, values_all = measure.marching_cubes(mask, 0.5)
      except ValueError:
          angle = 9999
          all_scans[file.stem] = angle
      else:

          #%%
          upper_angles = dict()
          lower_angles = dict()
          all_discs = dict()
          all_discs_string = dict()

          # label = 221
          for label in labels:
              total_discs += 1
              mask_temp = np.zeros_like(mask)
              mask_temp[mask == label] = 1
              current_disc = Disc(mask_temp)
              upper_angles[label] = current_disc.angle_U
              lower_angles[label] = current_disc.angle_L
              all_discs[label] = current_disc
              all_discs_string[str(label)] = current_disc.to_dict()

          #%% calculate largest Cobb angle
          overview = np.zeros([len(labels), len(labels)])

          for index_L, label_L in enumerate(labels):
              for index_U, label_U in enumerate(labels):
                  if label_L < label_U:
                      overview[index_L, index_U] = upper_angles[label_U] - lower_angles[label_L]

          overview = np.absolute(overview)
          if largest_angle is True:
              angle = overview.max()
              index_highest_vertebra, index_lowest_vertebra = np.unravel_index(overview.argmax(), overview.shape)
          else:
              index_highest_vertebra = int(overview.shape[0] - consensus[''][index] - 1)
              index_lowest_vertebra = int(overview.shape[0] - consensus['Dominique_bottom'][index])
              angle = overview[index_highest_vertebra, index_lowest_vertebra]
          all_scans[file.stem] = angle

          #%% plot 2D image
          mask_vertebrae[mask_vertebrae > 125] = 0
          mask_vertebrae[mask_vertebrae == 100] = 0
          mask_vertebrae[mask_vertebrae > 0] = 1
          image2D = mask_vertebrae.sum(1)
          fig = plt.figure()
          ax = fig.add_subplot()

          for label, disc in all_discs.items():
              x_L, z_L = plane_to_tangent(disc.verts, disc.point_L, disc.normal_L)
              x_U, z_U = plane_to_tangent(disc.verts, disc.point_U, disc.normal_U)
              color_L = 'white'
              linewidth_L = 0.5
              color_U = 'white'
              linewidth_U = 0.5
              if label == labels[index_highest_vertebra]:
                  color_L = 'red'
                  linewidth_L = 1
                  ax.plot([disc.point_L[0], disc.point_L[0] - 300 * disc.normal_L[0]],
                          [disc.point_L[2], disc.point_L[2] - 300 * disc.normal_L[2]], color=color_L, lw=1)
              elif label == labels[index_lowest_vertebra]:
                  color_U = 'red'
                  linewidth_U = 1
                  ax.plot([disc.point_U[0], disc.point_U[0] + 300 * disc.normal_U[0]],
                          [disc.point_U[2], disc.point_U[2] + 300 * disc.normal_U[2]], color=color_U, lw=1)
              ax.plot(x_L, z_L, color=color_L, linewidth=linewidth_L)
              ax.plot(x_U, z_U, color=color_U, linewidth=linewidth_U)

          # answer = ast.literal_eval(rows[index].answer)

          ax.imshow(image2D.T)
          ax.axis('off')
          ax.invert_yaxis()
          ax.invert_xaxis()

          plt.title(str('Cobb angle = ' + str(round(abs(angle)))))
          plt.show()
          fig.savefig(image_path / str(file.stem + '.png'))
          print(str(file.stem[0:17] + '.png'))
          index += 1

#%%

angles = pd.DataFrame(all_scans.values())
angles.to_excel(image_path / 'results.xlsx', index=False)

#%%

with open(image_path / 'results.csv', 'w') as f:
    w = csv.writer(f)
    for angle in angles_manual:
        w.writerow([round(angle, 2)])

