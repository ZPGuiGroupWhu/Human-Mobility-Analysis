// 管道函数：将上一次函数结果作为下一次的函数参数输入

export const pipe = (...fns) => initArg => {
  return fns.reduceRight((prev, fn) => {
    return fn(prev);
  }, initArg);
}