import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export class ImageNumber {
  baseName: string;
  start: number;
  length: number;
}

@Injectable({
  providedIn: 'root'
})
export class LoadService {
  private ImageNumberSource = new BehaviorSubject<ImageNumber>({
    baseName: 'assets/combine123/',
    start: 110000,
    length: 1024
  });
  imageNumber$ = this.ImageNumberSource.asObservable();

  reloadImage(imageNum: number) {
    const name = 'assets/combine123/';
    const begin = 110000;
    const offset = Math.floor(Math.random() * (10000 - imageNum));
    // console.log(offset);
    this.ImageNumberSource.next({
      baseName: name,
      start: begin + offset,
      length: imageNum
    });
  }

  constructor() {}

  get imageNumber() {
    return this.ImageNumberSource.getValue();
  }
}
