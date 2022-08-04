// 数组铺平
export const arrayFlat = (arr) => {
  return arr.reduce((prev, item) => {
    return Array.isArray(item) ? [...prev, ...arrayFlat(item)] : [...prev, item]
  }, [])
}