# -*- coding: utf-8 -*-
import unittest
from dataclasses import replace
import numpy as np
from pathlib import Path
from q4_solver import _fv_step,Config,load_terrain,initial_state,simulate
BASE=Path(__file__).resolve().parents[1]
class Q4Tests(unittest.TestCase):
    def test_lake_at_rest_well_balanced(self):
        z=np.ones((14,17))*5.;z[:,3:8]=7.;z[7:10,5:12]=8.
        h=10-z;q=np.zeros_like(h)
        hn,qn,rn=_fv_step(h,q,q,z,np.ones_like(h,dtype=np.bool_),2.,2.,.1)
        self.assertLess(float(abs(hn-h).max()),1e-9)
        self.assertLess(float(abs(qn).max()),1e-9)
        self.assertLess(float(abs(rn).max()),1e-9)
    def test_source_volume_and_dimensions(self):
        p=Config(grid_m=120,duration_s=30)
        T=load_terrain(BASE/'data'/'elevation_utm45n.tif',p)
        h,qx,qy,mask=initial_state(T,p)
        self.assertGreater(mask.sum(),15)
        self.assertAlmostEqual(float(h.sum()*T.dx*T.dy),p.initial_volume_m3,delta=1e-5)
        self.assertGreater(h.max(),30)
        self.assertLess(h.max(),100)
    def test_mass_conservation_no_erosion(self):
        p=Config(grid_m=120,duration_s=40,snapshot_s=20)
        T=load_terrain(BASE/'data'/'elevation_utm45n.tif',p)
        r=simulate(T,p)
        self.assertGreaterEqual(r['h'].min(),0.)
        self.assertLess(abs(r['records'][-1]['物质量平衡相对误差']),2e-6)
    def test_erosion_budget(self):
        p=Config(grid_m=120,duration_s=40,snapshot_s=20,mu=.08,xi_mps2=1800,erosion_k=.003,
                deposit_k_per_s=.0005)
        T=load_terrain(BASE/'data'/'elevation_utm45n.tif',p)
        r=simulate(T,p)
        self.assertGreaterEqual(r['h'].min(),0.)
        self.assertLess(abs(r['records'][-1]['物质量平衡相对误差']),2e-6)
        self.assertTrue(np.all(r['eroded']<=p.erodible_depth_m+1e-8))
if __name__=='__main__':unittest.main()
