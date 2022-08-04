export const eventEmitter = new class EventEmitter {
  static VERSION = '1.0.0'

  constructor() {
    this.store = {}
  }

  on(event, fn) {
    try {
      if (!event || !fn) throw new Error('缺少参数');
      if (this.isArray(event)) {
        event.forEach(item => {
          this.on(item, fn);
        })
      } else {
        (this.store[event] || (this.store[event] = [])).push(fn);
      }
    } catch (err) {
      console.log(err);
    }
    return this;
  }

  emit(event, ...args) {
    try {
      const fn = this.store[event];
      if (!fn) throw new Error('事件不存在');
      fn.forEach(item => {
        item.call(this, ...args);
      })
    } catch (err) {
      console.log(err);
    }
    return this;
  }

  off(event, fn) {
    if (!event || !this.store[event]) return this;
    if (!fn) {
      Reflect.deleteProperty(this.store, event);
    } else {
      for (let i=0; i<this.store[event].length; i++) {
        if ((this.store[event][i] === fn) || (this.store[event][i].fn === fn)) {
          this.store[event].splice(i, 1)
        }
      }
    }
    return this;
  }

  once(event, fn) {
    const vm = this;
    function rm(...args) {
      fn.call(vm, ...args);
      this.off(event, fn);
    }
    rm.fn = fn;
    this.on(event, rm);
    return this;
  }


  isArray(val) {
    if (!Array.isArray) {
      return Object.prototype.toString.call(val) === '[object Array]'
    } else {
      return Array.isArray(val)
    }
  }
}()