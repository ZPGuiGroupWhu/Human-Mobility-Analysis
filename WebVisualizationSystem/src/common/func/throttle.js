// 节流
export const throttle = (fn, interval) => {
  let timer = null;
  return function (...args) {
    let isOpen = !timer;
    const ctx = this;
    if (isOpen) {
      timer = setTimeout(() => {
        clearTimeout(timer);
        timer = null;
      }, interval)
      fn.call(ctx, ...args)
    }
  }
}