// 判断传入参数是否存在空字符：是则返回 false
export const isEmptyString = (...args) => {
  return args.reduce((prev, cur) => {
    // console.log((prev === false) ? false : prev && (cur.length !== 0));
    return (prev === false) ? false : prev && (cur.length !== 0);
  }, true)
}