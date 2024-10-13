import mapping_utils as mapping_utils
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import random
import torch
from torch.utils.data import Dataset
from collections import namedtuple
import os

DATASET_MEAN = 239.74347588572797 / 4096
DATASET_STD = 397.1364638124688 / 4096

DATASET_MEAN_T1_mapping = 197.62839 / 4096
DATASET_STD_T1_mapping = 333.961248 / 4096

DATASET_MEAN_T2_mapping = 396.6454446984587 / 4096
DATASET_STD_T2_mapping = 546.4103438777042 / 4096

TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = [3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]

SYN_PATIENTS = [181, 186, 84, 1748, 1812, 1821, 62, 1665, 196, 1978, 95, 57, 1750, 1708, 1773, 1622, 153, 1575, 1629, 109, 99, 1830, 1833, 1601, 1737, 1984, 127, 1584, 1860, 47, 18, 1814, 111, 1805, 221, 1613, 1603, 1587, 65, 1585, 1661, 1900, 1621, 120, 1699, 1614, 1963, 1715, 1726, 165, 171, 119, 1938, 1691, 170, 60, 5, 202, 1924, 1880, 1962, 1712, 1572, 143, 1969, 1616, 1793, 93, 1632, 1914, 1850, 1672, 64, 1677, 1829, 1893, 1878, 1945, 1890, 1970, 1656, 24, 73, 192, 1756, 1867, 105, 1954, 1999, 9, 158, 1883, 124, 1692, 1711, 1927, 1671, 1820, 1961, 1733, 1780, 1611, 1943, 206, 203, 1940, 1932, 38, 63, 1740, 1992, 1874, 173, 1871, 1903, 1816, 61, 1591, 1991, 76, 1818, 1885, 1669, 13, 115, 1869, 1845, 201, 1828, 1875, 72, 194, 1879, 1610, 1837, 174, 1870, 1925, 1653, 1789, 1735, 141, 184, 1634, 1644, 1888, 1997, 1567, 53, 1785, 178, 1695, 1721, 1668, 1645, 1619, 1933, 1976, 1977, 79, 1929, 42, 1660, 118, 218, 1559, 161, 1864, 97, 1817, 166, 1889, 1823, 1971, 133, 1965, 1784, 1956, 1998, 1792, 1936, 51, 142, 179, 195, 1844, 136, 112, 1982, 52, 1895, 1898, 1836, 1771, 1946, 39, 1994, 15]
SYN_ROUND2_PATIENTS = [1750, 1587, 103, 1918, 147, 66, 1647, 1657, 1791, 121, 1672, 1807, 1621, 1616, 125, 1640, 1635, 113, 129, 198, 1839, 59, 1995, 1565, 1667, 1656, 1632, 1909, 1786, 1980, 1915, 1711, 140, 1796, 1558, 1824, 1659, 1908, 8, 1603, 211, 1674, 116, 1599, 1843, 1780, 1803, 1793, 1592, 1792, 31, 1931, 1790, 1809, 1695, 1794, 83, 160, 1876, 49, 1739, 1990, 1951, 1746, 150, 1775, 1701, 117, 1586, 1694, 167, 67, 1618, 1863, 152, 1594, 1920, 1610, 1646, 1905, 1593, 1832, 108, 1953, 1851, 1958, 189, 1669, 1804, 1685, 1772, 1649, 1736, 1770, 1882, 1678, 1650, 1948, 1569, 32, 20, 1812, 77, 1620, 100, 1584, 1815, 1708, 1725, 169, 1767, 214, 1789, 2]#, 1922, 96, 154, 1740, 1666, 1993, 1859, 1638, 1765, 1857, 1645, 1641, 222, 144, 1588, 88, 1825, 1636, 1847, 27, 6, 1712, 217, 1887, 215, 1781, 1853, 1967, 1614, 1892, 1622, 1625, 1709, 1576, 138]
SYN_PATIENTS = SYN_PATIENTS + SYN_ROUND2_PATIENTS

SYN_PATIENTS = [id for id in SYN_PATIENTS if id not in [1736, 1770, 1678, 1650, 1569, 1812, 1620, 1584, 1708, 1725, 1767, 1789, 1740, 1666, 1638, 1765, 1645, 1641, 1588, 1636, 1712, 1781, 1614, 1622, 1625, 1709, 1576]]


TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS and id not in SYN_PATIENTS]

DIAG_SEG_23_08_TRAIN = [1584, 1731, 1712, 1639, 1714, 1625, 1758, 1612, 1661, 1700, 1668, 1788, 1784, 1793, 1649, 1735, 1654, 1603, 1671, 1613, 1657, 1568, 1705, 1642, 1767, 1794, 1771, 1592, 1707, 1647, 1804, 1739, 1650, 1799, 1640, 1748, 1616, 1696, 1627, 1695, 1681, 1609, 1738, 1744, 1672, 1809, 1567, 1734, 1798, 1750, 1772, 1646, 1608, 1765, 1723, 1756, 1593, 1641, 1716, 1770, 1620, 1599, 1789, 1774, 1753, 1763, 1591, 1638, 1601, 1619, 1733, 1644, 1787, 1699, 1722, 1812, 1569, 1571, 1682, 1736, 1785, 1648, 1690, 1803, 1651, 1791, 1691, 1576, 1718, 1615, 1807, 1623, 1725, 1667, 1602, 1761, 1796, 1617, 1587, 1800, 1562, 1678, 1780, 1776, 1610, 1680, 1595, 1709, 1762, 1683, 1577, 1582, 1653, 1792, 1583, 1692, 1754, 1730, 1594, 1811, 1775, 1559, 1560, 1614, 1656, 1708, 1808, 1740, 1563, 1726, 1635, 1677, 1564, 1636, 1589, 1628, 1729, 1660, 1757, 1766, 1674, 1605, 1606, 1783, 1732, 1751, 1578, 1622, 1694, 1795, 1659, 1728, 1769, 1773, 1720, 1581, 1621, 1645, 1721, 1611, 1643, 1713, 1777, 1597, 1669, 1565, 1684, 1666, 1598, 1588, 1633, 1790, 1737, 1686, 1711, 1604, 1685, 1801, 1629, 1632, 1760, 1806, 1586, 1805, 1558, 1585, 1746, 1580, 1575, 1786, 1600, 1752, 1572, 1781, 1665, 1634, 1715, 1673, 1573, 1579, 1693, 1701, 1618, 1727]
DIAG_SEG_23_08_VAL = [1652, 1561, 1719, 1557, 1566, 1658, 1630, 1698, 1688, 1675, 1687, 1706, 1607, 1782, 1574, 1702, 1747, 1717, 1743, 1810, 1778, 1697, 1664, 1745, 1663, 1689, 1703, 1704, 1802, 1624, 1797, 1590, 1631, 1779, 1768, 1676, 1596, 1742, 1570, 1655, 1759, 1741, 1662, 1749, 1724, 1637, 1626, 1755, 1764, 1710, 1670, 1679]

DIAG_SEG_23_08_TRAIN = [id for id in DIAG_SEG_23_08_TRAIN if id not in SYN_PATIENTS]

TRAIN_FROM_23 = [id for id in range(1813, 2000) if id not in SYN_PATIENTS]
TEST_FROM_23 = [id for id in range(2000, 2084+1)]

# The patients below are corrected in 2023_08_15, therefore they are ignored in 2023_07_09
CORRECTED = ['2023_07_09/Patient (1757)', '2023_07_09/Patient (1758)', '2023_07_09/Patient (1759)', '2023_07_09/Patient (1760)', '2023_07_09/Patient (1761)', '2023_07_09/Patient (1762)', '2023_07_09/Patient (1763)', '2023_07_09/Patient (1764)', '2023_07_09/Patient (1765)', '2023_07_09/Patient (1766)', '2023_07_09/Patient (1767)', '2023_07_09/Patient (1768)', '2023_07_09/Patient (1769)', '2023_07_09/Patient (1770)', '2023_07_09/Patient (1771)', '2023_07_09/Patient (1772)', '2023_07_09/Patient (1773)', '2023_07_09/Patient (1774)', '2023_07_09/Patient (1775)', '2023_07_09/Patient (1776)', '2023_07_09/Patient (1777)', '2023_07_09/Patient (1778)', '2023_07_09/Patient (1779)', '2023_07_09/Patient (1780)', '2023_07_09/Patient (1781)', '2023_07_09/Patient (1782)', '2023_07_09/Patient (1783)', '2023_07_09/Patient (1784)', '2023_07_09/Patient (1785)', '2023_07_09/Patient (1786)', '2023_07_09/Patient (1787)', '2023_07_09/Patient (1788)', '2023_07_09/Patient (1789)', '2023_07_09/Patient (1790)', '2023_07_09/Patient (1791)', '2023_07_09/Patient (1792)', '2023_07_09/Patient (1793)', '2023_07_09/Patient (1794)', '2023_07_09/Patient (1795)', '2023_07_09/Patient (1796)', '2023_07_09/Patient (1797)', '2023_07_09/Patient (1798)', '2023_07_09/Patient (1799)', '2023_07_09/Patient (1800)', '2023_07_09/Patient (1801)', '2023_07_09/Patient (1802)', '2023_07_09/Patient (1803)', '2023_07_09/Patient (1804)', '2023_07_09/Patient (1805)', '2023_07_09/Patient (1806)', '2023_07_09/Patient (1807)', '2023_07_09/Patient (1808)', '2023_07_09/Patient (1809)', '2023_07_09/Patient (1810)', '2023_07_09/Patient (1811)', '2023_07_09/Patient (1812)', '2023_07_09/Patient (1813)', '2023_07_09/Patient (1814)', '2023_07_09/Patient (1815)', '2023_07_09/Patient (1816)', '2023_07_09/Patient (1817)', '2023_07_09/Patient (1818)', '2023_07_09/Patient (1819)', '2023_07_09/Patient (1820)', '2023_07_09/Patient (1821)', '2023_07_09/Patient (1822)', '2023_07_09/Patient (1823)', '2023_07_09/Patient (1824)', '2023_07_09/Patient (1825)', '2023_07_09/Patient (1826)', '2023_07_09/Patient (1827)', '2023_07_09/Patient (1828)', '2023_07_09/Patient (1829)', '2023_07_09/Patient (1830)', '2023_07_09/Patient (1831)', '2023_07_09/Patient (1832)', '2023_07_09/Patient (1833)', '2023_07_09/Patient (1834)', '2023_07_09/Patient (1835)', '2023_07_09/Patient (1836)', '2023_07_09/Patient (1837)', '2023_07_09/Patient (1838)', '2023_07_09/Patient (1839)', '2023_07_09/Patient (1840)', '2023_07_09/Patient (1841)', '2023_07_09/Patient (1842)', '2023_07_09/Patient (1843)', '2023_07_09/Patient (1844)', '2023_07_09/Patient (1845)', '2023_07_09/Patient (1846)', '2023_07_09/Patient (1847)', '2023_07_09/Patient (1848)', '2023_07_09/Patient (1849)', '2023_07_09/Patient (1850)', '2023_07_09/Patient (1851)', '2023_07_09/Patient (1852)', '2023_07_09/Patient (1853)', '2023_07_09/Patient (1854)', '2023_07_09/Patient (1855)', '2023_07_09/Patient (1856)', '2023_07_09/Patient (1857)', '2023_07_09/Patient (1858)', '2023_07_09/Patient (1859)', '2023_07_09/Patient (1860)', '2023_07_09/Patient (1861)', '2023_07_09/Patient (1862)', '2023_07_09/Patient (1863)', '2023_07_09/Patient (1864)', '2023_07_09/Patient (1865)', '2023_07_09/Patient (1866)', '2023_07_09/Patient (1867)', '2023_07_09/Patient (1868)', '2023_07_09/Patient (1869)', '2023_07_09/Patient (1997)', '2023_07_09/Patient (1998)', '2023_07_09/Patient (1999)', '2023_07_09/Patient (2000)', '2023_07_09/Patient (2001)', '2023_07_09/Patient (2002)', '2023_07_09/Patient (2003)', '2023_07_09/Patient (2004)', '2023_07_09/Patient (2005)', '2023_07_09/Patient (2006)', '2023_07_09/Patient (2007)', '2023_07_09/Patient (2008)', '2023_07_09/Patient (2009)', '2023_07_09/Patient (2010)', '2023_07_09/Patient (2011)', '2023_07_09/Patient (2012)', '2023_07_09/Patient (2013)', '2023_07_09/Patient (2014)', '2023_07_09/Patient (2015)', '2023_07_09/Patient (2016)', '2023_07_09/Patient (2017)', '2023_07_09/Patient (2018)', '2023_07_09/Patient (2019)', '2023_07_09/Patient (2020)', '2023_07_09/Patient (2021)', '2023_07_09/Patient (2022)', '2023_07_09/Patient (2023)', '2023_07_09/Patient (2024)', '2023_07_09/Patient (2025)', '2023_07_09/Patient (2026)', '2023_07_09/Patient (2027)', '2023_07_09/Patient (2028)', '2023_07_09/Patient (2029)', '2023_07_09/Patient (2030)', '2023_07_09/Patient (2031)', '2023_07_09/Patient (2032)', '2023_07_09/Patient (2033)', '2023_07_09/Patient (2034)', '2023_07_09/Patient (2035)', '2023_07_09/Patient (2036)', '2023_07_09/Patient (2037)', '2023_07_09/Patient (2038)', '2023_07_09/Patient (2039)', '2023_07_09/Patient (2040)', '2023_07_09/Patient (2041)', '2023_07_09/Patient (2042)', '2023_07_09/Patient (2043)', '2023_07_09/Patient (2044)', '2023_07_09/Patient (2045)', '2023_07_09/Patient (2046)', '2023_07_09/Patient (2047)', '2023_07_09/Patient (2048)', '2023_07_09/Patient (2049)', '2023_07_09/Patient (2050)', '2023_07_09/Patient (2051)', '2023_07_09/Patient (2052)', '2023_07_09/Patient (2053)', '2023_07_09/Patient (2054)', '2023_07_09/Patient (2055)', '2023_07_09/Patient (2056)', '2023_07_09/Patient (2057)', '2023_07_09/Patient (2058)', '2023_07_09/Patient (2059)', '2023_07_09/Patient (2060)', '2023_07_09/Patient (2061)', '2023_07_09/Patient (2062)', '2023_07_09/Patient (2063)', '2023_07_09/Patient (2064)', '2023_07_09/Patient (2065)', '2023_07_09/Patient (2066)', '2023_07_09/Patient (2067)', '2023_07_09/Patient (2068)', '2023_07_09/Patient (2069)', '2023_07_09/Patient (2070)', '2023_07_09/Patient (2071)', '2023_07_09/Patient (2072)', '2023_07_09/Patient (2073)', '2023_07_09/Patient (2074)', '2023_07_09/Patient (2075)', '2023_07_09/Patient (2076)', '2023_07_09/Patient (2077)', '2023_07_09/Patient (2078)', '2023_07_09/Patient (2079)', '2023_07_09/Patient (2080)', '2023_07_09/Patient (2081)', '2023_07_09/Patient (2082)', '2023_07_09/Patient (2083)', '2023_07_09/Patient (2084)']

IGNORED_SAMPLES = [
    "2022_11_22/Patient (5)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (10)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (49)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (49)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (40)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (71)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (104)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (93)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (72)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (77)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (73)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (120)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (119)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (111)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (72)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (120)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (111)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (218)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (218)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (365)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (335)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (356)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (365)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (319)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (306)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (444)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (416)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (390)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (424)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (416)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (390)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (444)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (424)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1655)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1658)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1661)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1661)/T1_Mapping_/Mid.dcm",
    "2023_07_09/Patient (1661)/T2_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1663)/T2_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1663)/T1_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1870)/T1_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1870)/T1_Mapping_/Base.dcm",
    "2023_08_15/Patient (1870)/T1_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294586246366594.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294587301366595.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294587768466596.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294588249266597.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294588723566598.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294589190666599.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294589666066600.dcm",
    "2023_08_15/Patient (1870)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715294590130066601.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294577398466525.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294578524966560.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294579003966563.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294579483166566.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294579960866569.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294580418866572.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294580877866575.dcm",
    "2023_08_15/Patient (1870)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020715294581324366578.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294581787566582.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294582866566585.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294583332166588.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294583803266589.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294584274266590.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294584743366591.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294585317766592.dcm",
    "2023_08_15/Patient (1870)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715294585792866593.dcm",
    "2023_08_15/Patient (1870)/T2_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1870)/T2_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1870)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715271866832665854.dcm",
    "2023_08_15/Patient (1870)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715271866889365856.dcm",
    "2023_08_15/Patient (1870)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020715271866940865858.dcm",
    "2023_08_15/Patient (1870)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715271866668465848.dcm",
    "2023_08_15/Patient (1870)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715271866722265850.dcm",
    "2023_08_15/Patient (1870)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020715271866779765852.dcm",
    "2023_08_15/Patient (1871)/T1_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1871)/T1_Mapping_/Base.dcm",
    "2023_08_15/Patient (1871)/T1_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342926849892736.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342927948392737.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342928437292738.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342928971692739.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342929491392740.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342929994092741.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342930514492742.dcm",
    "2023_08_15/Patient (1871)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114342931000192743.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342917607492667.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342918740792702.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342919238192705.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342919744192708.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342920223592711.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342920712792714.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342921201892717.dcm",
    "2023_08_15/Patient (1871)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022013114342921711692721.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342922209392724.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342923289392727.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342923807592730.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342924331192731.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342924843392732.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342925343792733.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342925859992734.dcm",
    "2023_08_15/Patient (1871)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114342926356592735.dcm",
    "2023_08_15/Patient (1871)/T2_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1871)/T2_Mapping_/Base.dcm",
    "2023_08_15/Patient (1871)/T2_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1871)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114323319337192526.dcm",
    "2023_08_15/Patient (1871)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114323319391292528.dcm",
    "2023_08_15/Patient (1871)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022013114323319444092530.dcm",
    "2023_08_15/Patient (1871)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022013114323319007392514.dcm",
    "2023_08_15/Patient (1871)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022013114323319061992516.dcm",
    "2023_08_15/Patient (1871)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022013114323319121092518.dcm",
    "2023_08_15/Patient (1871)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114323319173192520.dcm",
    "2023_08_15/Patient (1871)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114323319229092522.dcm",
    "2023_08_15/Patient (1871)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022013114323319283892524.dcm",
    "2023_08_15/Patient (1872)/T1_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1872)/T1_Mapping_/Base.dcm",
    "2023_08_15/Patient (1872)/T1_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474592252528804.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474593202128805.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474593619328806.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474594032728807.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474594438528808.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474594859728809.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474595262128810.dcm",
    "2023_08_15/Patient (1872)/T1_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211474595664528811.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474584595328735.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474585567428770.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474585990328773.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474586403528776.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474586811028779.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474587224028782.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474587627128785.dcm",
    "2023_08_15/Patient (1872)/T1_map_base/1.3.12.2.1107.5.2.18.142101.2022020211474588046028789.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474588450028792.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474589385928795.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474589805728798.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474590217428799.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474590633128800.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474591038228801.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474591448528802.dcm",
    "2023_08_15/Patient (1872)/T1_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211474591859028803.dcm",
    "2023_08_15/Patient (1872)/T2_Mapping_/Apex.dcm",
    "2023_08_15/Patient (1872)/T2_Mapping_/Base.dcm",
    "2023_08_15/Patient (1872)/T2_Mapping_/Mid.dcm",
    "2023_08_15/Patient (1872)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211455830077628594.dcm",
    "2023_08_15/Patient (1872)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211455830133528596.dcm",
    "2023_08_15/Patient (1872)/T2_map_apex/1.3.12.2.1107.5.2.18.142101.2022020211455830194228598.dcm",
    "2023_08_15/Patient (1872)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022020211455829746128582.dcm",
    "2023_08_15/Patient (1872)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022020211455829800828584.dcm",
    "2023_08_15/Patient (1872)/T2_map_base/1.3.12.2.1107.5.2.18.142101.2022020211455829855628586.dcm",
    "2023_08_15/Patient (1872)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211455829910228588.dcm",
    "2023_08_15/Patient (1872)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211455829964128590.dcm",
    "2023_08_15/Patient (1872)/T2_map_mid_/1.3.12.2.1107.5.2.18.142101.2022020211455830022928592.dcm",
] + CORRECTED


class SeDataset(Dataset):

    def __init__(self,
            data_root,
            resolution,
            classes=3,
            random_crop=False,
            random_flip=False,
            split='train',
            type_labeling=False,
            resize=False):
        super().__init__()


        self.transforms = None#transforms
        self.split = split
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.resolution = resolution
        self.type_labeling = type_labeling
        self.num_classes = classes
        self.resize = resize

        contours_filename = f"Contours.json"

        self.all_samples = mapping_utils.construct_samples_list(
            data_root, contours_filename
        )
        mapping_utils.print_diagnostics(data_root, self.all_samples)

        partitions = {
            "train": TRAIN_PATIENTS + DIAG_SEG_23_08_TRAIN + TRAIN_FROM_23,
            "val": VAL_PATIENTS + DIAG_SEG_23_08_VAL,
            "test": TEST_PATIENTS + TEST_FROM_23,
            "syn": SYN_PATIENTS,
        }

        print(len(self.all_samples))

        self.all_samples = self.remove_ignored(self.all_samples, IGNORED_SAMPLES, data_root)


        self.samples = mapping_utils.split_samples_list(
            self.all_samples, partitions[self.split]
        )

        self.to_mapping_only()

        print(f"Number of samples: {len(self.samples)}")


    # we only need the mapping images
    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict: (sample, mask)
        """
        path, mask_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None, use_modality_lut=False)
        mask_contours = mapping_utils.load_contours(mask_path)
        mask = mapping_utils.contours_to_masks(mask_contours, sample.shape)

        size_ori = sample.shape

        if self.transforms is not None:           
            transformed = self.transforms(image=sample, mask=mask)
            sample, mask = transformed["image"], transformed["mask"]

        if 'T2' in path and self.type_labeling:            
            mask = mask + self.num_classes
        
        sample = sample / 4095
        if self.resize:
            # Convert images to channels_first mode, from albumentations' 2d grayscale images
            sample = resize(sample, (self.resolution, self.resolution), anti_aliasing=True)
            
            mask = resize(mask, (self.resolution, self.resolution), anti_aliasing=False, mode='edge', preserve_range=True, order=0)
        else:            
            # fill the image with zeros to make it resolution x resolution
            sample = np.pad(sample, ((0, self.resolution - sample.shape[0]), (0, self.resolution - sample.shape[1])), mode='constant', constant_values=0)
            mask += 1
            mask = np.pad(mask, ((0, self.resolution - mask.shape[0]), (0, self.resolution - mask.shape[1])), mode='constant', constant_values=0)

        sample = np.expand_dims(sample, 0)

        mask = np.expand_dims(mask, 0)

        sample = sample.astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.random_crop and random.random() < 0.33:
            opt = {
                'preprocess': 'resize_and_crop',      
                'crop_size': int(self.resolution),
                'load_size': int(self.resolution * 1.5),
                'flip': self.random_flip,
            }
            
            transform_params = get_params(opt, sample.shape[1:])
            B_transform = get_transform(opt, transform_params, method=transforms.InterpolationMode.NEAREST, grayscale=True)
            A_transform = get_transform(opt, transform_params, grayscale=True)
                      
            sample = torch.from_numpy(sample)
            mask = torch.from_numpy(mask)
            
            sample = A_transform(sample)
            mask = B_transform(mask)

            sample = sample.numpy()
            mask = mask.numpy()
        
        mask = np.squeeze(mask, axis=0)
  
        if 'T1' in path:
            sample = (sample - DATASET_MEAN_T1_mapping) / (DATASET_STD_T1_mapping)
        elif 'T2' in path:
            sample = (sample - DATASET_MEAN_T2_mapping) / (DATASET_STD_T2_mapping)


        # normalize the sample to -1, 1
        sample = (sample - np.min(sample.flatten())) / (np.max(sample.flatten()) - np.min(sample.flatten()))

        # mask to torch tensor
        mask = torch.from_numpy(mask).long()
        sample = torch.from_numpy(sample).float()

        out_dict = {}
        out_dict['path'] = path
        out_dict['label_ori'] = mask.detach().clone()
        out_dict['label'] = mask[None,]
        out_dict['size_ori'] = size_ori
        out_dict['mask_path'] = mask_path


        return sample, out_dict

    def __len__(self) -> int:
        return len(self.samples)


    @staticmethod
    def remove_ignored(samples, ignore_list, dataset_root, verbose=False):
        samples_numpy = np.array(samples)
        remove_count = 0
        for sample in ignore_list:
            sample = os.path.join(dataset_root, sample)  
            # sample is like     "2022_11_22/Patient (5)/T1_Mapping_/Apex.dcm" 
            # map sample is like "2022_11_22/Patient (5)/T1_Mapping_/_map_apex/"
            # also sample can be a Patient folder, e.g.: "2022_11_22/Patient (5)"
            map_sample = sample.replace("_Mapping_/Apex.dcm", "_map_apex")
            map_sample = map_sample.replace("_Mapping_/Mid.dcm", "_map_mid_")
            map_sample = map_sample.replace("_Mapping_/Base.dcm", "_map_base")
            for s in [sample, map_sample]:
                samples_to_drop = np.char.startswith(samples_numpy[:,0], s)
                if samples_to_drop.any():
                    samples_numpy = samples_numpy[~samples_to_drop]
                    if verbose:
                        print(colored(f"Removed {s} from the dataset based on the IGNORED_SAMPLES list", "red"))
                    remove_count += np.sum(samples_to_drop)
        print(f"Removed {remove_count} samples from the dataset based on the IGNORED_SAMPLES list")
        return samples_numpy.tolist()




def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['load_size']
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w

    x = random.randint(0, np.maximum(0, new_w - opt['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_size']))


    return {'crop_pos': (x, y)}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    #if grayscale:
    #    transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt['preprocess']:
        osize = [opt['load_size'], opt['load_size']]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt['preprocess']:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['crop_size'])))

    if opt['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.shape[1:]
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img[:, y1:y1 + th, x1:x1 + tw]
    return img

