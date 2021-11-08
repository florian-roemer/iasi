import os
import sys
import numpy as np
import time
# generic EPS library
os.environ["EUGENE_HOME"] = os.environ["HOME"] + "/.local/share/eugene"
import eugene


class data:
    """
    Class: func:`data` of IASI level 1c reader

    How to use (very brief):
    1) initialise class with a IASI L1c file
    2) call read_spectra with the mdr number as variable
    3) call get_spectra with ifov and pixel number as input
        repeat step 3 for every spectra desired from the given mdr
   4) call read_geoloc for an mdr, reads all lat/lon pairs for every pixel
      (ifov*pn --> 120) of mdr into lat/lon filed
      repeat step4 for every mdr desired
    """

    def __init__(self, IasiFile, openOption='r', giadr_sf_read=True,
                 mdr=None):
        """
        init the :func:`data`class

        :param str IasiFile: IASI file name
        :param str openOption: optional - open option. Default is 'r'
        :param boolean giadr_sf_read: optional - read giadr scale factors.
               Default is 'True'
        :rtype: object
        """

        self.IasiFile = IasiFile
        self.L1c_data = eugene.Product(self.IasiFile, openOption)

        if mdr is None:
            self.nbMDR = self.L1c_data.get("mdr-1c.!items")
            self.MDR = np.arange(self.nbMDR)
        else:
            self.nbMDR = len(mdr)
            self.MDR = mdr

        self._spectra = []
        self.wavenumber = []
        self.spectra_orbit = np.zeros(shape=(self.nbMDR, 8461, 30, 4))

        self.lat = []
        self.lon = []

        self.lat_orbit = np.zeros(shape=(self.nbMDR, 120))
        self.lon_orbit = np.zeros(shape=(self.nbMDR, 120))

        self.IDefSpectDWn1b = None
        self.IDefNsfirst1b = None
        self.IDefNslast1b = None

        self.IDefScaleSondNbScale = 0
        self.IDefScaleSondNsfirst = None
        self.IDefScaleSondNslast = None
        self.IDefScaleSondScaleFactor = None

        if giadr_sf_read:
            self._read_giadr_sf()

        # read instrument angles
        self.GGeoSondAnglesMETOP = np.array(
            [self.L1c_data.get("mdr-1c[%d].GGeoSondAnglesMETOP" % mdr)
             for mdr in self.MDR])

        # read cloud fraction
        self.GEUMAvhrr1BCldFrac = np.array(
            [self.L1c_data.get("mdr-1c[%d].GEUMAvhrr1BCldFrac" % mdr)
             for mdr in self.MDR])

        # read land fraction
        self.GEUMAvhrr1BLandFrac = np.array(
            [self.L1c_data.get("mdr-1c[%d].GEUMAvhrr1BLandFrac" % mdr)
             for mdr in self.MDR])

    def _read_giadr_sf(self):
        self.IDefScaleSondNbScale = self.L1c_data.get(
                'giadr-scalefactors.IDefScaleSondNbScale')
        self.IDefScaleSondNsfirst = self.L1c_data.get(
                'giadr-scalefactors.IDefScaleSondNsfirst')
        self.IDefScaleSondNslast = self.L1c_data.get(
                'giadr-scalefactors.IDefScaleSondNslast')
        self.IDefScaleSondScaleFactor = self.L1c_data.get(
                'giadr-scalefactors.IDefScaleSondScaleFactor')

    def calc_wavenumber(self, mdr):
        self.IDefSpectDWn1b = (self.L1c_data.get(
                "mdr-1c[%d].IDefSpectDWn1b" % mdr))
        self.IDefNsfirst1b = (self.L1c_data.get(
                "mdr-1c[%d].IDefNsfirst1b" % mdr))
        self.IDefNslast1b = (self.L1c_data.get(
                "mdr-1c[%d].IDefNslast1b" % mdr))

        self.wavenumber = []

        for k in range(self.IDefNslast1b - self.IDefNsfirst1b + 1):
            self.wavenumber.append(
                    self.IDefSpectDWn1b * (self.IDefNsfirst1b + k - 2)/100)

    def get_wavenumber(self):
        return self.wavenumber

    def read_spectra(self, mdr):
        self._spectra = (self.L1c_data.get("mdr-1c[%d].GS1cSpect" % mdr))
        self.calc_wavenumber(mdr)

    def get_spectra(self, ifov, pn, apply_sf=True):
        spectra = np.array(self._spectra[ifov][pn])

        if apply_sf:
            spectra = self._apply_sf(spectra)

        return spectra

    def get_spec_orbit(self):
        # return one full spectra for the 4 IASI pixels
        # for the full orbit (=x mdr)
        # spec[mdr,8461,4,30]

        for m, mdr in enumerate(self.MDR):
            sp = self.read_spectra(mdr)
            for ip in range(4):
                for ifov in range(30):
                    self.spectra_orbit[m, :, ifov, ip] = \
                        self.get_spectra(ifov, ip)

    def _apply_sf(self, spectra):
        spectra_final = []
        for x in range(self.IDefScaleSondNbScale):
            sp_tmp = spectra[self.IDefScaleSondNsfirst[x] -
                             self.IDefNsfirst1b:self.IDefScaleSondNslast[x] -
                             self.IDefNsfirst1b + 1] *\
                             10**(-1*self.IDefScaleSondScaleFactor[x])
            spectra_final = np.append(spectra_final, sp_tmp)
        return spectra_final

    def read_geoloc(self, mdr):
        lat = []
        lon = []
        geoloc = (self.L1c_data.get("mdr-1c[%d].GGeoSondLoc" % mdr))

        for ifov in range(len(geoloc)):
            for pn in range(len(geoloc[ifov])):
                lat.append(geoloc[ifov][pn][1])
                lon.append(geoloc[ifov][pn][0])

        self.lat = lat
        self.lon = lon

    def get_orbit_lat_lon(self):
        for m, mdr in enumerate(self.MDR):
            self.read_geoloc(mdr)
            self.lat_orbit[m, :] = np.array(self.lat)
            self.lon_orbit[m, :] = np.array(self.lon)


if __name__ == '__main__':
    start = time.process_time()

    mdr = np.concatenate((np.arange(50, 250), np.arange(430, 630)))
    mydata = data('/mnt/lustre02/work/um0878/data/iasi/iasi-l1/reprocessed/'
                  'm02/2017/07/15/'
                  'IASI_xxx_1C_M02_20170715174155Z_20170715192059Z_N_O_'
                  '20170715192029Z',
                  mdr=mdr)

    mydata.get_orbit_lat_lon()
    mydata.get_spec_orbit()
    print("Lon shape: ", mydata.lon_orbit.shape)
    print("Lat shape: ", mydata.lat_orbit.shape)
    print("Data shape: ", mydata.spectra_orbit.shape)
    end = time.process_time()
    print(end - start)
