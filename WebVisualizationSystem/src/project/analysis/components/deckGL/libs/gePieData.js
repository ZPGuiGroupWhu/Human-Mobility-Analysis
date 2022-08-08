// 同级格子里poi的类别和数量
export const getPieData = (info) => {
  const cellCenter = info?.object?.position;
  const originData = info?.object?.points;
  const data = originData?.reduce((prev, { source }) => {
    const { typeId, type } = source;
    if (!Reflect.has(prev, typeId)) {
      Reflect.set(prev, typeId, {
        cellCenter,
        typeId,
        type,
        value: 1,
      });
    } else {
      Reflect.set(prev[typeId], "value", prev[typeId].value + 1);
    }
    return prev;
  }, {});
  return data ?? {};
};
