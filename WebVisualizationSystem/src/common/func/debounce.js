// 防抖
export const debounce = (fn, wait) => {
  let timer = null;
  return function (...args) {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }

    const ctx = this;
    timer = setTimeout(() => {
      fn.call(ctx, ...args);
    }, wait)

  }
}