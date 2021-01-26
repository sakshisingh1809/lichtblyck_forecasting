module "k_w5core.dll"
include "\\lbfiler3\apps\Tools\Energiewirtschaft\Kisters\BelVis\bas\KiBasic_Lib_include.bas"
w5login("PFMSTROM", "s1K1uUZ78hcb", "BELVIS.LICHTBLICK.DE")

integer vb_values
string sql
string basepath = "c:\temp\memoized\timeseries_"

float ts_from, ts_until

integer id = 1850102 
ts_from = timestamp(2021, 1, 1, 0, 0, 0, 0)
ts_until = timestamp(2021, 1, 2, 0, 0, 0, 0)

vb_values = _w5gettimeseries(id, wintertime(ts_from), wintertime(ts_until))
vbgotop(vb_values)
vb_values = _vbremovefields(vb_values, "date,time")
vbexport(vb_values, basepath + stri(id) + ".csv", "|", 1,-1, 1,-1)

_vbclose(vb_values)