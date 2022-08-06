/**
 * 基于给定字段判断唯一性，并生成该字段值的 "唯一数组" 
 * @param array 
 * @param callback 
 * @returns 
 */
export const getUniqueByKey = (array, callback) => {
  return array.reduce((prev, cur, idx, arr) => {
    const target = callback(cur, idx, arr);
    if (!prev.includes(target)) {
      prev.push(target);
    }
    return prev;
  }, []);
};
