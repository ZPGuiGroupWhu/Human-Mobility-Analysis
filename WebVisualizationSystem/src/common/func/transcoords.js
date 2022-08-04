import gcoord from 'gcoord';

export default function transcoords (arrs) {
  if (Array.isArray(arrs[0])) {
    return arrs.map(arr =>
      gcoord.transform(
        arr,
        gcoord.WGS84,
        gcoord.BD09
      )
    );
  } else {
    return gcoord.transform(
      arrs,
      gcoord.WGS84,
      gcoord.BD09
    )
  }
}