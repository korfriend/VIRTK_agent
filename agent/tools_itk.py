def itk_gaussian(in_img: str, out_img: str, sigma: float=1.0) -> dict:
    import itk
    img = itk.imread(in_img)
    out = itk.discrete_gaussian_image_filter(img, variance=sigma**2)
    itk.imwrite(out, out_img)
    return {"path": out_img}
