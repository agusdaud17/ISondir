# ISondir

ISondir merupakan module python yang digunakan untuk melakukan korelasi data pengujian sondir (CPT), menjadi profil tanah, dan parameter tanah.

## Fitur
1. Averaging (filtering) data mentah (qc dan fs).
2. Penentuan SBT zona menggunakan 2 metode yaitu:
    - Dengan nilai index Ic
    - Dengan color RGB; secara default menggunakan color RGB file SBTlow.jpg
3. Plot grafik Robertson 2010
4. Profil lapisan tanah.
5. Export parameter ke format CSV.
6. Estimasi parameter tanah
    - Berat isi tanah $(\gamma)$ kN/m3
    - Sudut geser internal $(\phi)$ derajat
    - Kuat geser undrained $(S_u)$ kPa
    - Ekivalen N-SPT $(N_{60})$
    - Kepadatan relatif $(D_r)$ persen (%)
    - *Overconsolidated ratio* $(OCR)$
    - Modulus constraint $(M)$ kPa
    - Modulus elastisitas $(E)$ kPa
    - *Small-strain modulus* $(G_0)$ kPa
    - *Shear wave velocity* $(V_s)$
    - Rigidity index $(I_G)$
    - Normalized rigidity index $(K_G)$
7. Digunakan bersama module **IFondasi** untuk menghitung kapasitas dan penurunan pondasi dangkal.
8. Dapat melakukan tiga (3) kategori kalkulasi untuk:
    - Single point (satu titik pengujian)
    - Segment (beberapa data pengujian)
    - Multi-segment (beberapa segment)

## Input data
Input data yang diperlukan, berupa **format excel** terdiri dari tiga kolom data yaitu:
1. Tahanan konus (qc), satuan default MPa.
2. Tahanan selimut (fs), satuan default kPa.
3. Tekanan air pori (u2), satuan default kPa.

Untuk lebih jelas, dapat dilihat pada contoh file: raw_data.xlsx di atas.

## Text Editor

Untuk menggunakan module ini direkomendasikan menggunakan **Jupyter Notebook/ Jupyter Lab**.

## Contoh
Berikut contoh perintah minimum untuk menggunakan module ISondir
```
s1 = ISondir.Single(id_='CPT-01')

s1.read_excel('raw_data.xlsx')

# Muka air tanah, contoh kedalaman 1.0 m
s1.setGWL(z=1)

# Melakukan kakulasi parameter
s1.solve_basic()
s1.solve_parameters()

```

## Plot Grafik

Plot grafik Soil Behavior Type (SBT), dan profil lapisan tanah.
```
s1.raw_plot()
s1.basic_plot()
s1.plot_SBTn()
s1.plot_rigidityIndex()
s1.soil_profil()
s1.plot_norm()
```

Plot hasil estimasi parameter tanah ke dalam grafik.
```
s1.estimation_plot1()
s1.estimation_plot2()
s1.estimation_plot3()
```

# Contoh Tampilan Plot

![SBT chart](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj4FOuE4HqWU6NtkH8lN_ug0w8jxuiwxOYdz-p02T5XX6Rdw7FCKQBXeWS6Hw58azV6ACyEAw43asOpcwytUyPNv4moelIf9PFC6-3pwtN5ifrltDAzjIdA33dnNtDtuUmszzGbxRDmvYgXtzV3k-INCF--7ZX5Unm-y8qV17u0KlP-mz8voB9DZyhOFA/s544/plot_SBTn.png)

![Soil Profil](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhmPe80YNZGg1XXdZnR5ce6mlHkvEdESV_xRJFuzs8JjB-uT0OqvxURgP2osQdMRRsh7_J87giAKroBThbw7CyOldbaAxgDOG-JMhj6g-lJasFVS_z-nk8S1NsoVLx8VQjg21qD2ZL55qU5c6Y0kWVEXKBC2iKteK_licXJLaaq4t9k26VEm0bwPZN3Zw/w321-h400/soil_profil.png)

![Estimation plot 1](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjEtoxetmbtp_C7C_I13QzqoLlAw6Fi_iqX90vwP5gVTDKwzD8SgLXgX6UUBC2XCWQENlHNKZZVJ3jkXdEoQox6w1Ui5DnXUcQ3odmFvoNWt9V24Ynpt4QRh8rlLovJ8ZE2Cl0uxLxHdgzEzLWk6VtNIKCrjRSJKTy_gErERbK5xLehZjFkSFv4mPDYMQ/w400-h351/est_1.png)

![SBTN](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhJjC5TecusMoTmHnfZMG57spvoQ-Gv8OjzgAUAtSipxGu1jx5A-lf2P-2-LxWlo5ClDu0yjeRbutSpGFmqbkshO6VywjP5mx5k0G0d1h_dvLStl8m-QC_5LtK_pdKacccjMK1nx6_ce-gi_za6zMM-pe3InQTCK3UClsL3EyKSKQ0FPeeT9UYkSUytgg/w400-h363/plot_norm.png)
